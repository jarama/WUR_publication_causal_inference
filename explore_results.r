# Load libraries
library(data.table)
library(ggplot2)

# Load data
results <- fread("data/processed/extracted/1_sim.csv")
results[, `:=` (n = factor(n),
                beta_0 = round(beta_0, digits = 2),
                beta_1 = round(beta_1, digits = 2),
                beta_2 = round(beta_2, digits = 2),
                beta_3 = round(beta_3, digits = 2),
                beta_4 = round(beta_4, digits = 2),
                beta_5 = round(beta_5, digits = 2),
                beta_6 = round(beta_6, digits = 2),
                beta_7 = round(beta_7, digits = 2),
                beta_8 = round(beta_8, digits = 2),
                sd_Y = factor(round(sd_Y, digits = 2)),
                direct_effect = round(direct_effect, digits = 2),
                indirect_effect = round(indirect_effect, digits = 2),
                total_effect = round(total_effect, digits = 2),
                Y_sd = factor(round(Y_sd, digits = 2)))]

### Visualize violin plots

# n,beta_0,beta_1
dedup <- unique(
  results[incl_vars == "('T',)" & Y_sd == "1", 
          .(n, beta_0, beta_1, direct_effect, indirect_effect, total_effect)]
)
ggplot(
  results[incl_vars == "('T',)" & Y_sd == "1", ],
  aes(x = n, y = coef_T)
) +
  geom_violin() +
  geom_point(data = dedup, aes(x = n, y = direct_effect),
             color = "green", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = n, y = indirect_effect),
             color = "red", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = n, y = total_effect),
             color = "yellow", size = 3, inherit.aes = FALSE) +
  facet_grid(beta_1 ~ beta_0, 
             labeller = label_both) +
  labs(title = "incl_vars == ('T',) & Y_sd == 1")

# Y_sd,beta_0,beta_1
dedup <- unique(
  results[incl_vars == "('T',)" & n == "100", 
          .(Y_sd, beta_0, beta_1, direct_effect, indirect_effect, total_effect)]
)
ggplot(
  results[incl_vars == "('T',)" & n == "100", ],
  aes(x = Y_sd, y = coef_T)
) +
  geom_violin() +
  geom_point(data = dedup, aes(x = Y_sd, y = direct_effect),
             color = "green", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = Y_sd, y = indirect_effect),
             color = "red", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = Y_sd, y = total_effect),
             color = "yellow", size = 3, inherit.aes = FALSE) +
  facet_grid(beta_1 ~ beta_0, 
             labeller = label_both) +
  labs(title = "incl_vars == ('T',) & n == 100")

# incl_vars,beta_0,beta_1
dedup <- unique(
  results[n == "100" & Y_sd == "1", 
          .(incl_vars, beta_0, beta_1, direct_effect, indirect_effect, total_effect)]
)
ggplot(
  results[n == "100" & Y_sd == "1", ],
  aes(x = incl_vars, y = coef_T)
) +
  geom_violin() +
  geom_point(data = dedup, aes(x = incl_vars, y = direct_effect),
             color = "green", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = incl_vars, y = indirect_effect),
             color = "red", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = incl_vars, y = total_effect),
             color = "yellow", size = 3, inherit.aes = FALSE) +
  facet_grid(beta_1 ~ beta_0, 
             labeller = label_both) +
  labs(title = "n == 100 & Y_sd == 1")

# incl_vars,n,Y_sd
dedup <- unique(
  results[beta_0==1 & beta_1==1, 
          .(incl_vars, n, Y_sd, direct_effect, indirect_effect, total_effect)]
)
ggplot(
  results[beta_0==1 & beta_1==1, ],
  aes(x = incl_vars, y = coef_T)
) +
  geom_violin() +
  geom_point(data = dedup, aes(x = incl_vars, y = direct_effect),
             color = "green", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = incl_vars, y = indirect_effect),
             color = "red", size = 3, inherit.aes = FALSE) +
  geom_point(data = dedup, aes(x = incl_vars, y = total_effect),
             color = "yellow", size = 3, inherit.aes = FALSE) +
  facet_grid(n ~ Y_sd, 
             labeller = label_both) +
  labs(title = "beta_0 == 1 & beta_1 == 1")
