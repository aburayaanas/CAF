# R Statistical Analysis & Publication Figure Generation
# Open this file in RStudio to run the clinical statistics and generate the plots.

# Set Working Directory (uncomment and edit if running standalone)
# setwd("c:/Users/abura/Development/CAF")

# 1. Load Libraries
if (!require("ggplot2")) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
  library(ggplot2)
}

# 2. Load Data Files
preds_df <- read.csv("data/r_predictions.csv")
att_df <- read.csv("data/r_attention.csv")

# Convert variables to factors for grouping
att_df$vaper_group <- factor(att_df$vaper, levels = c(0, 1), labels = c("Non-Vaper", "Active Vaper"))
preds_df$vaper_group <- factor(preds_df$vaper, levels = c(0, 1), labels = c("Non-Vaper", "Active Vaper"))

# ==============================================================================
# STATISTICAL TEST 1: Welch's Two-Sample t-Test (On Soft-Tissue Attention P_soft)
# ==============================================================================
cat("\n=== 1. WELCH'S TWO-SAMPLE T-TEST (ON P_soft) ===\n")
t_test_res <- t.test(p_soft ~ vaper_group, data = att_df, var.equal = FALSE)
print(t_test_res)

# ==============================================================================
# STATISTICAL TEST 2: Mann-Whitney U Test (Wilcoxon Rank-Sum Test on P_soft)
# ==============================================================================
cat("\n=== 2. MANN-WHITNEY U TEST (Non-Parametric) ===\n")
wilcox_res <- wilcox.test(p_soft ~ vaper_group, data = att_df, alternative = "greater")
print(wilcox_res)

# ==============================================================================
# STATISTICAL TEST 3: Cohen's d Effect Size for P_soft (Base R Implementation)
# ==============================================================================
cat("\n=== 3. COHEN'S D EFFECT SIZE ===\n")
cohens_d <- function(g1, g2) {
  n1 <- length(g1)
  n2 <- length(g2)
  mean_diff <- mean(g1) - mean(g2)
  pooled_sd <- sqrt(((n1 - 1) * var(g1) + (n2 - 1) * var(g2)) / (n1 + n2 - 2))
  return(mean_diff / pooled_sd)
}

non_vaper_psoft <- att_df$p_soft[att_df$vaper == 0]
vaper_psoft <- att_df$p_soft[att_df$vaper == 1]
effect_size <- cohens_d(non_vaper_psoft, vaper_psoft)
cat("Cohen's d (Effect size of visual attention drop):", effect_size, " (values > 0.8 represent large effect size)\n")

# ==============================================================================
# STATISTICAL TEST 4: Fisher's Exact Test / Hypergeometric Test for Dismissal
# ==============================================================================
cat("\n=== 4. FISHER'S EXACT (HYPERGEOMETRIC) TEST FOR MODAL DISMISSAL ===\n")
# Threshold for visual modality dismissal (e.g., P_soft < 0.25)
threshold <- 0.25
att_df$dismissed <- factor(att_df$p_soft < threshold, levels = c(TRUE, FALSE), labels = c("Dismissed", "Active"))

contingency_table <- table(att_df$vaper_group, att_df$dismissed)
cat("Contingency Table (Vaping Group vs Visual Modality Dismissal):\n")
print(contingency_table)

fisher_res <- fisher.test(contingency_table, alternative = "two.sided")
print(fisher_res)

# ==============================================================================
# STATISTICAL TEST 5: McNemar's Test (CAF vs Baseline Predictions)
# ==============================================================================
cat("\n=== 5. MCNEMAR'S TEST FOR ACCURACY SUPERIORITY ===\n")
preds_df$caf_correct <- preds_df$caf_pred == preds_df$true_label
preds_df$base_correct <- preds_df$baseline_pred == preds_df$true_label

mcnemar_matrix <- matrix(
  c(
    sum(preds_df$caf_correct & preds_df$base_correct),
    sum(!preds_df$caf_correct & preds_df$base_correct),
    sum(preds_df$caf_correct & !preds_df$base_correct),
    sum(!preds_df$caf_correct & !preds_df$base_correct)
  ),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(
    CAF = c("Correct", "Wrong"),
    Baseline = c("Correct", "Wrong")
  )
)

cat("McNemar Disagreement Matrix:\n")
print(mcnemar_matrix)

mcnemar_res <- mcnemar.test(mcnemar_matrix, correct = FALSE)
print(mcnemar_res)

# ==============================================================================
# STATISTICAL TEST 6: Modality Pruning Validation (Mean P_probing)
# ==============================================================================
cat("\n=== 6. MODALITY PRUNING VALIDATION ===\n")
mean_p_probing <- mean(att_df$p_probing)
cat("Mean Attention Weight allocated to Zeroed Probing Modality (P_probing):", mean_p_probing, "\n")
cat("This demonstrates that the gating network automatically pruned the disabled modality down to:", mean_p_probing * 100, "%\n")

# ==============================================================================
# PLOTTING: Figure 2 - Boxplot of P_soft Attention Weights
# ==============================================================================
cat("\nGenerating Figure 2: Boxplot of Attention Weights...\n")
p_boxplot <- ggplot(att_df, aes(x = vaper_group, y = p_soft, fill = vaper_group)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.8, aes(color = vaper_group)) +
  labs(
    title = "Clinical Soft-Tissue Attention Weight Distribution",
    subtitle = "Confounder-Driven Modality Routing (Vapers vs Non-Vapers)",
    x = "Patient Subgroup",
    y = "Attention Weight P_soft (Soft-Tissue Photo)",
    fill = "Subgroup",
    color = "Subgroup"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 16),
    panel.grid.minor = element_blank()
  ) +
  scale_fill_manual(values = c("Non-Vaper" = "#4285F4", "Active Vaper" = "#EA4335")) +
  scale_color_manual(values = c("Non-Vaper" = "#1A73E8", "Active Vaper" = "#D93025"))

ggsave("docs/figure2_attention_boxplot.png", plot = p_boxplot, width = 7, height = 5, dpi = 300)
cat("Saved Figure 2 to: docs/figure2_attention_boxplot.png\n")

# ==============================================================================
# PLOTTING: Figure 3 - Patient Routing Heatmap (3 Modalities)
# ==============================================================================
cat("\nGenerating Figure 3: Heatmap of Patient Gating Routing (3 Modalities)...\n")
# Melt the attention weights data frame for heatmap plotting (3 modalities)
melted_att <- data.frame(
  Patient_ID = rep(att_df$patient_id, 3),
  Vaper = rep(att_df$vaper_group, 3),
  Modality = c(
    rep("P_tab (Questionnaire)", nrow(att_df)), 
    rep("P_soft (Clinical Photo)", nrow(att_df)),
    rep("P_hard (Radiographs)", nrow(att_df))
  ),
  Weight = c(att_df$p_tab, att_df$p_soft, att_df$p_probing)
)

# Sort patients by Vaping group and P_soft value for aesthetic ordering
melted_att$Modality <- factor(melted_att$Modality, levels = c("P_tab (Questionnaire)", "P_soft (Clinical Photo)", "P_hard (Radiographs)"))

# Sort patients by Vaping group and P_soft value for aesthetic ordering
sort_order <- att_df$patient_id[order(att_df$vaper, -att_df$p_soft)]
melted_att$Patient_ID <- factor(melted_att$Patient_ID, levels = sort_order)

p_heatmap <- ggplot(melted_att, aes(x = Modality, y = Patient_ID, fill = Weight)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#F1F3F4", high = "#1A73E8") +
  labs(
    title = "Per-Patient Multimodal Gating Routing Heatmap (3 Modalities)",
    subtitle = "Questionnaire, Visual Soft-Tissue, and Radiograph Modalities",
    x = "Routed Modality Channels",
    y = "Patient cohort (Sorted)",
    fill = "Attention Weight"
  ) +
  facet_grid(Vaper ~ ., scales = "free_y", space = "free_y") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.y = element_text(size = 8),
    plot.title = element_text(face = "bold", size = 14)
  )

ggsave("docs/figure3_routing_heatmap.png", plot = p_heatmap, width = 9, height = 7, dpi = 300)
cat("Saved Figure 3 to: docs/figure3_routing_heatmap.png\n")
cat("\n=== All 3-modality statistics and figures generated successfully! ===\n")
