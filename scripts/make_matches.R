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

# Run matching for early initiation group
message("Running matching for early initiation group...")
early_match <- matching_pipeline(
    input_path = config$early,
    output_dir = "./models/matching/",
    method = "full",
    distance = "glm",
    prefix = "early",
    save_datasets = TRUE
)

write_matched_data(early_match, filename = "early_matched.parquet")
write_love_plot(early_match, filename = "early_love_plot.png")

message("Early matching complete:")
message("  - Treated: ", sum(early_match$treat))
message("  - Controls: ", sum(early_match$treat == 0))
message("  - Matched units: ", sum(early_match$weights > 0))

# Run matching for late initiation group
message("\nRunning matching for late initiation group...")
late_match <- matching_pipeline(
    input_path = config$late,
    output_dir = "./models/matching/",
    method = "full",
    distance = "glm",
    prefix = "late",
    save_datasets = TRUE
)

write_matched_data(late_match, filename = "late_matched.parquet")
write_love_plot(late_match, filename = "late_love_plot.png")

message("Late matching complete:")
message("  - Treated: ", sum(late_match$treat))
message("  - Controls: ", sum(late_match$treat == 0))
message("  - Matched units: ", sum(late_match$weights > 0))

message("\nAll matching complete.")
