##-----------------------------------------------------------------------------#
#' Process results 
#' 
#' Take the .csv and .log output from an experiment and process it into useful
#' data frames summarizing the important model metrics.
##-----------------------------------------------------------------------------#

library(tidyverse)
library(here)
library(glue)
library(ggbeeswarm)

param_files <- here("application/params") %>% 
  list.files(full.names = TRUE) 

experiments <- 
  param_files %>% 
  set_names(nm = str_replace_all(param_files, "(.*)/experiment-(.*).csv", "\\2")) %>% 
  map(read_csv) %>% 
  bind_rows(.id = "experiment")

# strange conversion from None to nan
experiments$file <- str_replace_all(experiments$file, "None", "nan")

## Performance metrics per run ------------------------------------------------#

try_read_csv <- function(file, ...) {
  if (file.exists(file)) {
    return(read_csv(file, ...))
  } else {
    return(NULL)
  }
}

tmp <- experiments %>% 
  rowwise() %>% 
  mutate(
    results_dir = glue("results/tma/"),
    results = list(
      try_read_csv(here(results_dir, file), show_col_types = FALSE)
    )
  ) %>% 
  filter(!is.null(results))

df_metrics <- tmp %>% 
  hoist(
    results,
    loss = "loss",
    mae = "mae",
    acc = "accuracy"
  ) %>% 
  rowwise() %>% 
  mutate(
    mil_pool_combo = glue("{mil_method}-{pooling_mode}")
  )

# Save output
df_metrics %>% 
  select(-results) %>% 
  write_csv("application/results/tma_test_metrics.csv")

## Results from parameter choice ----------------------------------------------#

df_gridsearch <- 
  experiments %>% 
  mutate(file = str_replace_all(file, "_metrics.csv", "_gridsearch_summary\\.csv")) %>% 
  rowwise() %>% 
  mutate(
    mil_pool_combo = glue("{mil_method}-{pooling_mode}"),
    results_dir = glue("results/tma/"),
    results = list(
      try_read_csv(here(results_dir, file), show_col_types = FALSE)
    )
  ) %>% 
  filter(!is.null(results)) %>% 
  unnest_longer(results) %>% 
  tidyr::unpack(results)

write_csv(df_gridsearch, "application/results/tma_gridsearch_summary.csv")

## Attention weights ----------------------------------------------#

df_att <- 
  experiments %>% 
  mutate(file = str_replace_all(file, "_metrics.csv", "_att-weights\\.csv")) %>% 
  rowwise() %>% 
  mutate(
    mil_pool_combo = glue("{mil_method}-{pooling_mode}"),
    results_dir = glue("results/tma/"),
    results = list(
      try_read_csv(here(results_dir, file), show_col_types = FALSE, col_names = FALSE)
    )
  ) %>% 
  filter(!is.null(results)) %>% 
  unnest_longer(results) %>% 
  tidyr::unpack(results)

write_csv(df_att, "application/results/tma_attention-weights.csv")


## Long-format training iteration progress ------------------------------------#

df_train_prog <-
  experiments %>% 
  mutate(file = str_replace_all(file, "_metrics.csv", "_training\\.log")) %>% 
  rowwise() %>% 
  mutate(
    mil_pool_combo = glue("{mil_method}-{pooling_mode}"),
    results_dir = glue("results/tma/"),
    results = list(
      try_read_csv(here(results_dir, file), show_col_types = FALSE)
    )
  ) %>% 
  filter(!is.null(results)) %>% 
  unnest_longer(results) %>% 
  tidyr::unpack(results)

write_csv(df_train_prog, "application/results/tma_train-val_progression.csv")


