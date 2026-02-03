library(yaml)
library(arrow)
library(dplyr)
library(MatchIt)
library(cobalt)
library(ggplot2)


#' Load data for matching
#'
#' @param fpath Path to the parquet file
#' @param treatment_col Column name for treatment indicator (default: "initiation_group")
#' @param treatment_value Value indicating treatment group (default: "never" indicates control)
#' @return Data frame prepared for matching
load_matching_data <- function(fpath, treatment_col = "initiation_group",
                                treatment_value = "never") {
    df <-
        arrow::read_parquet(fpath) %>%
        mutate(treatment = ifelse(.data[[treatment_col]] == treatment_value, 0, 1)) %>%
        mutate(across(where(~ is.character(.x)), as.factor)) %>%
        select(where(~ !is.factor(.x) || nlevels(.x) >= 2)) %>%
        as.data.frame()

    if ("participant_id" %in% names(df)) {
        rownames(df) <- df$participant_id
    }

    df
}



#' Get matched pairs from MatchIt output
#'
#' For 1:1 matching methods, returns case-control pairs.
#' For full matching, returns subclass assignments with weights.
#'
#' @param match_out MatchIt object
#' @return Data frame with matching information
get_matched_pairs <- function(match_out) {
    # Full matching doesn't have match.matrix - use subclass/weights instead
    if (is.null(match_out$match.matrix)) {
        matched_data <- MatchIt::match.data(match_out)
        return(
            matched_data %>%
                select(any_of(c("subclass", "weights", "treatment"))) %>%
                tibble::rownames_to_column(var = "id")
        )
    }

    # 1:1 matching - return pairs
    match_out$match.matrix %>%
        as.data.frame() %>%
        tibble::rownames_to_column(var = "case") %>%
        rename(c("match" = "V1")) %>%
        tidyr::drop_na()
}


#' Get balance summary from MatchIt output
#'
#' @param match_out MatchIt object
#' @return Data frame with standardized mean differences
get_balance_summary <- function(match_out) {
    balance_table <- summary(match_out, standardize = TRUE)$sum.matched
    as.data.frame(balance_table) %>%
        tibble::rownames_to_column("variable") %>%
        filter(variable != "distance")
}


#' Write matched data to disk
#'
#' Extracts matched data from a MatchIt object and writes to parquet format.
#'
#' @param match_out MatchIt object
#' @param filename Output filename (without path)
#' @param output_dir Output directory (default: "data/processed")
#' @return Invisibly returns the matched data frame
write_matched_data <- function(match_out, filename = "matched_data.parquet",
                               output_dir = "./data/processed") {
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    matched_df <- MatchIt::match.data(match_out)
    output_path <- file.path(output_dir, filename)

    arrow::write_parquet(matched_df, output_path)
    message("Wrote matched data to: ", output_path)

    invisible(matched_df)
}


#' Run propensity score matching
#'
#' @param data Data frame prepared for matching
#' @param method Matching method (nearest, optimal, full, etc.)
#' @param distance Distance metric (glm, mahalanobis, euclidean, etc.)
#' @param exclude_cols Columns to exclude from matching formula
#' @param ... Additional arguments passed to matchit
#' @return MatchIt object
run_matching <- function(data, method = "nearest", distance = "glm",
                          exclude_cols = c("participant_id", "initiation_group",
                                           "initiation_timepoint", "treatment"),
                          ...) {
    covariate_cols <- setdiff(names(data), exclude_cols)
    match_formula <- as.formula(paste(
        "treatment ~",
        paste(covariate_cols, collapse = " + ")
    ))

    matchit(
        match_formula,
        data = data,
        method = method,
        distance = distance,
        ...
    )
}


#' Save matching results
#'
#' @param match_out MatchIt object
#' @param output_dir Directory for output files
#' @param prefix Prefix for output filenames
#' @param save_datasets Whether to save matched data and pairs as parquet
write_matching_results <- function(match_out, output_dir = "./models/matching/",
                                    prefix = "match", save_datasets = FALSE) {
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    if (save_datasets) {
        matched_data <- match.data(match_out)
        matched_pairs <- get_matched_pairs(match_out)
        balance_summary <- get_balance_summary(match_out)

        arrow::write_parquet(
            matched_data,
            file.path(output_dir, paste0(prefix, "_matched.parquet"))
        )

        arrow::write_parquet(
            matched_pairs,
            file.path(output_dir, paste0(prefix, "_pairs.parquet"))
        )

        arrow::write_parquet(
            balance_summary,
            file.path(output_dir, paste0(prefix, "_balance.parquet"))
        )
    }

    saveRDS(match_out, file.path(output_dir, paste0(prefix, "_model.rds")))
}


#' Create love plot for balance visualization
#'
#' @param match_out MatchIt object
#' @param output_path Path to save the plot (optional)
#' @param order Variable ordering ("alphabetical" or "adjusted")
#' @param threshold SMD threshold for balance (default: 0.1)
#' @return ggplot object
make_love_plot <- function(match_out, output_path = NULL,
                            order = "adjusted", threshold = 0.1) {
    p <- cobalt::love.plot(
        match_out,
        abs = TRUE,
        thresholds = c(m = threshold),
        var.order = order,
        position = "top",
        limits = c(0, 1)
    )

    if (!is.null(output_path)) {
        ggplot2::ggsave(plot = p, filename = output_path, dpi = "retina")
    }

    p
}


#' Write love plot to disk
#'
#' Creates a love plot showing covariate balance and saves to reports/figures.
#'
#' @param match_out MatchIt object
#' @param filename Output filename (default: "love_plot.png")
#' @param output_dir Output directory (default: "reports/figures")
#' @param width Plot width in inches (default: 8)
#' @param height Plot height in inches (default: 10)
#' @param threshold SMD threshold for balance (default: 0.1)
#' @param order Variable ordering ("alphabetical" or "adjusted")
#' @return Invisibly returns the ggplot object
write_love_plot <- function(match_out, filename = "love_plot.png",
                            output_dir = "./reports/figures",
                            width = 8, height = 10,
                            threshold = 0.1, order = "adjusted") {
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    p <- make_love_plot(match_out, order = order, threshold = threshold)
    output_path <- file.path(output_dir, filename)

    ggplot2::ggsave(
        plot = p,
        filename = output_path,
        width = width,
        height = height,
        dpi = "retina"
    )
    message("Wrote love plot to: ", output_path)

    invisible(p)
}


#' Full matching pipeline
#'
#' Convenience function that runs the full matching workflow
#'
#' @param input_path Path to input parquet file
#' @param output_dir Directory for outputs
#' @param method Matching method
#' @param distance Distance metric
#' @param prefix Output file prefix
#' @param exclude_patterns Regex patterns for columns to exclude from matching
#' @param save_datasets Whether to save matched data as parquet
#' @return MatchIt object
matching_pipeline <- function(input_path, output_dir = "./models/matching/",
                               method = "nearest", distance = "glm",
                               prefix = NULL,
                               exclude_patterns = NULL,
                               save_datasets = FALSE) {
    # Load data
    data <- load_matching_data(input_path)

    # Optionally exclude columns matching patterns
    if (!is.null(exclude_patterns)) {
        for (pattern in exclude_patterns) {
            data <- data %>% select(-matches(pattern))
        }
    }

    # Run matching
    match_out <- run_matching(data, method = method, distance = distance)

    # Generate prefix if not provided
    if (is.null(prefix)) {prefix <- paste(method, distance, sep = "_")}

    # Save results
    write_matching_results(match_out, output_dir, prefix, save_datasets)

    match_out
}
