#!/usr/bin/env Rscript
#
# Run propensity score matching for early and late initiation groups
#

# Install optmatch if not available (required for full matching, not on conda-forge)
if (!requireNamespace("optmatch", quietly = TRUE)) {
    message("Installing optmatch package...")
    install.packages("optmatch", repos = "https://cloud.r-project.org", quiet = TRUE)
}

source("R/matching.R")

# Load configuration
config <- yaml::read_yaml("conf/filepaths.yaml")

run_and_save <- function(label, input_path, prefix, treatment_col = "initiation_group") {
    message("\nRunning matching for ", label, "...")
    match_out <- matching_pipeline(
        input_path = input_path,
        output_dir = "./models/matching/",
        method = "full",
        distance = "glm",
        prefix = prefix,
        save_datasets = TRUE,
        treatment_col = treatment_col
    )

    write_matched_data(match_out, filename = paste0(prefix, "_matched.parquet"))
    write_love_plot(match_out, filename = paste0(prefix, "_love_plot.png"))

    message(label, " matching complete:")
    message("  - Treated: ", sum(match_out$treat))
    message("  - Controls: ", sum(match_out$treat == 0))
    message("  - Matched units: ", sum(match_out$weights > 0))

    match_out
}

early_match      <- run_and_save("early", config$early, "early")
late_match       <- run_and_save("late",  config$late,  "late")
early_never_match <- run_and_save("early never", config$early_never, "early_never")
late_never_match  <- run_and_save("late never",  config$late_never,  "late_never")

message("\nAll matching complete.")
