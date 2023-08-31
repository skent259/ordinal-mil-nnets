##-----------------------------------------------------------------------------#
#' Generate figures for paper
#' 
#' - Boxplots of results by ordinal method, mil method, and data set
#' - Tukey HSD plots for ordinal methods, mil methods
##-----------------------------------------------------------------------------#

library(tidyverse)
library(here)
library(glue)
library(patchwork)
library(emmeans)

source(here("analysis/utils-analysis.R"))

fig_dir <- "analysis/figures"
theme_set(theme_light())

# Pull in needed results
exp_list <- c("fgnet-1.0.1", "fgnet-1.0.2", "fgnet-1.0.3", "fgnet-1.0.4", 
              "bcnb-3.0.1", "bcnb-3.0.2", "amrev-4.0.1", "imdb-5.0.1", "afad-6.0.1")

df_metrics <- read_csv(here("results/test_metrics_from_experiments.csv")) %>% 
  filter(experiment %in% exp_list)
  
models <- readRDS(here("analysis/objects/models-01.rds"))

# Some re-usabable plotting components
y_or_scale <- scale_y_discrete(
  limits = methods_ordinal()$method,
  labels = methods_ordinal()$short_name
)
y_mil_scale <- scale_y_discrete(
  limits = methods_mil()$method,
  labels = methods_mil()$short_name
)
x_or_scale <- scale_x_discrete(
  limits = methods_ordinal()$method,
  labels = methods_ordinal()$short_name
)

## Two-way plot of main effects -----------------------------------------------#

plot_two_factors <- function(
    df, 
    factors = c("ordinal_method", "mil_pool_combo"),
    metric = "mae", 
    facets = "data_set_type", 
    ...) {
  
  df %>% 
    ggplot(aes_(
      x = as.name(factors[1]), 
      y = as.name(metric), 
      color = as.name(factors[2])
    )) +
    geom_boxplot(outlier.alpha = 0.4) +
    stat_summary(geom = "point", 
                 position = position_dodge2(width = 0.75),
                 fun = mean, size = 3, shape = 1) +
    facet_wrap(facets, scales="free_x", ...) + 
    theme(legend.position = "bottom") +
    coord_flip()
}

top_bottom_to_left_right <- c(1, 5, 2, 6, 3, 7, 4, 8)

plot_two_factor_custom <- function(metric = "mae") {
  df_metrics %>% 
    mutate(
      mil_pool_combo = fct_relevel(mil_pool_combo, methods_mil()$method),
      data_set_type = case_when(
        data_set_type == "AFAD" ~ "AFAD",
        data_set_type == "AMREV_TV" ~ "AMREV (TV)",
        data_set_type == "BCNB_ALN" ~ "BCNB",
        data_set_type == "FGNET" ~  "FGNET",
        data_set_type == "IMDB" ~ "IMDB",
      )
    ) %>% 
    plot_two_factors(metric = metric, nrow=1) +
    x_or_scale +
    scale_color_manual(
      limits = rev(methods_mil()$method)[top_bottom_to_left_right],
      labels = rev(methods_mil()$short_name)[top_bottom_to_left_right],
      values = rev(methods_mil()$color)[top_bottom_to_left_right]
    ) +
    labs(
      color = NULL,
      x = NULL,
      y = str_to_upper(metric)
    )
}

metrics <- set_names(c("mae", "mzoe", "rmse"))

p_two_factor <- imap(metrics, ~plot_two_factor_custom(.x))
walk(p_two_factor, print)
iwalk(p_two_factor, ~ggsave(here(fig_dir, glue("plot_two-factor-{.y}.pdf")), .x, width = 8, height = 8.75))

## Tukey HSD plots ------------------------------------------------------------#

plot_tukey_hsd <- function(emmean, metric) {
  plot(
    emmean, 
    comparisons=TRUE,
    colors = c("black", "grey50", "grey50", "#24436D")
  ) + 
    labs(
      y = NULL,
      x = glue("Estimated marginal mean for {str_to_upper(metric)} outcome")
    ) 
}

# Make all base plots
p_tukey_or <- map2(models$emmean_or_tukey, models$outcome, 
                   ~plot_tukey_hsd(.x, .y) + y_or_scale + ggtitle("Ordinal methods"))
p_tukey_mil <- map2(models$emmean_mil_tukey, models$outcome,
                    ~plot_tukey_hsd(.x, .y) + y_mil_scale + ggtitle("MIL+pooling methods"))
names(p_tukey_or) <- models$outcome
names(p_tukey_mil) <- models$outcome

# Combine with {patchwork}
p_tukey_mae <- p_tukey_or$mae / p_tukey_mil$mae
print(p_tukey_mae)
ggsave(here(fig_dir, "plot_tukey-mae-ordinal-mil.pdf"), p_tukey_mae, width = 8, height = 5)

p_tukey_mzoe <- p_tukey_or$mzoe / p_tukey_mil$mzoe
print(p_tukey_mzoe)
ggsave(here(fig_dir, "plot_tukey-mzoe-ordinal-mil.pdf"), p_tukey_mzoe, width = 8, height = 5)

p_tukey_rmse <- p_tukey_or$rmse / p_tukey_mil$rmse
print(p_tukey_rmse)
ggsave(here(fig_dir, "plot_tukey-rmse-ordinal-mil.pdf"), p_tukey_rmse, width = 8, height = 5)

