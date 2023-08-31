library(colorspace)

#' Create methods df for plotting
methods_ordinal <- function() {
  tibble::tribble(
    ~method, ~short_name, ~color,
    "CLM_QWK_CLOGLOG", "CLM QWK (cloglog)", "#1b9e77",
    "CLM_QWK_LOGIT", "CLM QWK (logit)", "#d95f02",
    "CLM_QWK_PROBIT", "CLM QWK (probit)", "#66a61e",
    "CORAL", "CORAL", "#e7298a",
    "CORN", "CORN", "#7570b3",
  )
}

#' #' Create methods df for plotting
#' methods_mil <- function() {
#'   # https://coolors.co/palette/471ca8-884ab2-ff930a-f24b04-d1105a
#'   # other pallete tried https://coolors.co/palette/422680-341671-280659-660f56-ae2d68-f54952
#'   tibble::tribble(
#'     ~method, ~short_name, ~color,
#'     "MI_NET-mean", "mi-net (mean)", "#D1105A", 
#'     "MI_GATED_ATTENTION-NA", "MI-net (gated attention)", "#884AB2", 
#'     "MI_NET-max", "mi-net (max)", darken("#D1105A"), 
#'     "MI_ATTENTION-NA", "MI-net (attention)", "#471CA8", 
#'     "CAP_MI_NET-mean", "MI-net (mean)", "#F24B04", 
#'     "CAP_MI_NET_DS-max", "MI-net (DS, max)", darken("#FF930A"), 
#'     "CAP_MI_NET-max", "MI-net (max)", darken("#F24B04"), 
#'     "CAP_MI_NET_DS-mean", "MI-net (DS, mean)", "#FF930A", 
#'   )
#' }

#' Create methods df for plotting
methods_mil <- function() {
  # https://coolors.co/palette/471ca8-884ab2-ff930a-f24b04-d1105a
  # other pallete tried https://coolors.co/palette/422680-341671-280659-660f56-ae2d68-f54952
  tibble::tribble(
    ~method, ~short_name, ~color,
    "MI_NET-mean", "mi-net (mean)", "#a6cee3", 
    "MI_GATED_ATTENTION-NA", "MI-net (gated attention)", "#cab2d6", 
    "MI_NET-max", "mi-net (max)", darken("#1f78b4"), 
    "MI_ATTENTION-NA", "MI-net (attention)", "#6a3d9a", 
    "CAP_MI_NET-mean", "MI-net (mean)", "#fdbf6f", 
    "CAP_MI_NET_DS-max", "MI-net (DS, max)", darken("#e31a1c"), 
    "CAP_MI_NET-max", "MI-net (max)", darken("#ff7f00"), 
    "CAP_MI_NET_DS-mean", "MI-net (DS, mean)", "#fb9a99", 
  )
}

plot_two_factors <- function(
    df, 
    exps, 
    factors = c("ordinal_method", "mil_pool_combo"),
    metric = "mae", 
    facets = NULL, ...) {
  
  df %>% 
    filter(experiment %in% exps) %>% 
    ggplot(aes_(
      x = as.name(factors[1]), 
      y = as.name(metric), 
      color = as.name(factors[2])
      # shape = as.name(factors[1])
    )) +
    geom_boxplot() +
    stat_summary(geom = "point", 
                 position = position_dodge2(width = 0.75),
                 fun = mean, size = 3, shape = 3) +
    facet_wrap(facets, scales="free_x", ...) + 
    scale_color_brewer(palette = "Dark2", na.value = "black") +
    theme_bw() + 
    guides(shape = "none") + 
    theme(legend.position = "bottom") + 
    coord_flip()
}
