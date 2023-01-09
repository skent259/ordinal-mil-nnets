#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

library(tidyverse)
library(here)
library(glue)
library(ggbeeswarm)
library(ggfortify)
library(patchwork)
library(plotly)

# library(lme4)
# library(emmeans)

# Read in data
df_metrics <- read_csv(here("results/test_metrics_from_experiments.csv"))
df_train_prog <- read_csv(here("results/train-val_progression_from_experiments.csv"))

exp_list <- c("fgnet-1.0.1", "fgnet-1.0.2", "fgnet-1.0.3", "fgnet-1.0.4", 
              "bcnb-3.0.1", "bcnb-3.0.2", "amrev-4.0.1", "imdb-5.0.1", "afad-6.0.1")

`%ni%` <- Negate(`%in%`)

train_metrics <- colnames(df_train_prog)[16:ncol(df_train_prog)]
# data_set_types <- unique(df_train_prog$data_set_type)
ordinal_methods <- unique(df_train_prog$ordinal_method)
mil_pool_combo_methods <- unique(df_train_prog$mil_pool_combo)



plot_training_progression <- function(df, metric, group, alpha = 0.1) {
  
  df %>% 
    ggplot(aes(
      x = epoch, 
      y = .data[[metric]],
      group = file, 
      color = {{ group }} 
    )) + 
    geom_line(alpha = alpha) +
    scale_y_log10() +
    guides(colour = guide_legend(override.aes = list(alpha = 1))) + 
    theme_minimal() +
    labs(color = "") +
    theme(legend.position = "bottom")
}

##-----------------------------------------------------------------------------#
## UI --------------------------------------------------------------------------
##-----------------------------------------------------------------------------#

input_selection <- tribble(
  ~id, ~label, ~choices,
  "experiments", "Experiment(s) to view", exp_list,
  # "data_sets", "Data set(s) to view", data_set_types,
  "ordinal_methods", "Ordinal method(s) to view", ordinal_methods,
  "mil_methods", "MIL method(s) to view", mil_pool_combo_methods
) %>% 
  rowwise() %>% 
  mutate(
    input_ui = list(shiny::selectizeInput(id, label, choices = choices, multiple = TRUE, selected = choices))
  )

ui <- fluidPage(
  
  # Application title
  titlePanel("Analyze Results"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      selectizeInput("metric",
                     "Metric to show",
                     train_metrics),
      input_selection$input_ui
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      # plotly::plotlyOutput("training_plot")
      plotOutput("training_plot", height = 800)
    )
  )
)

##-----------------------------------------------------------------------------#
## Server ----------------------------------------------------------------------
##-----------------------------------------------------------------------------#

server <- function(input, output) {
  
  output$training_plot <- renderPlot({
    req(input$metric, input$experiments, input$ordinal_methods, input$mil_methods)

    df <- df_train_prog %>%
      filter(experiment %in% input$experiments) %>%
      # filter(data_set_type %in% input$data_sets) %>%
      filter(ordinal_method %in% input$ordinal_methods) %>%
      filter(mil_pool_combo %in% input$mil_methods)

    print(df)
    
    p <- plot_training_progression(df, input$metric, group = mil_pool_combo) + 
      facet_wrap(~ordinal_method, scales = "free_y")
    # plotly::ggplotly(p)
    p
  })

}

# Run the application 
shinyApp(ui = ui, server = server)
