


suppressMessages({
  library(tidymodels)   
  library(tidyclust)    
  library(ggplot2)      
  library(cluster)      
  library(patchwork)    
  library(dplyr)
  library(tidyr)
  library(forcats)
  library(scales)
})

library(recipes)
library(parsnip)
library(tidymodels)
library(lubridate)
tidymodels::tidymodels_prefer()

theme_set(ggplot2::theme_grey(base_size = 13))

set.seed(81995)

dados_ml <- read.csv2("dados_ml.csv") 

dados_ml$dias <- yday(dados_ml$Epoca_de_semeadura)

dados <- dados_ml %>% 
  select(-c(COD_PROD, Cod_Estacao_Met, Area_colhida, Epoca_de_semeadura, Cultivar, Produtividade)) %>%
  drop_na() |> 
  dplyr::mutate(across(where(is.numeric), as.numeric),
                across(!where(is.numeric), as.factor))


  

# summary(dados$Produtividade)
View(teste)

skimr::skim(teste)


# 2) Recipe: remover y, checar ZV e normalizar --------------------------------
rec <- recipe(~ ., data = dados) |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_numeric_predictors())

prep_rec <- prep(rec)
X_norm_tbl <- bake(prep_rec, new_data = NULL)


# 3) Especificação k-means e workflow -----------------------------------------
k_spec <- k_means(num_clusters = tune()) |>
  set_engine("stats", algorithm = "Hartigan-Wong", nstart = 20)

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(k_spec)

# 4) Reamostragem, grade de k e métricas --------------------------------------
folds <- vfold_cv(dados, v = 5)
grade <- grid_regular(num_clusters(range = c(2L, 8L)), levels = 7)
mets  <- cluster_metric_set(sse_within_total, sse_total, sse_ratio)

#> Within-Cluster Sum of Squares (WSS): Soma das distâncias quadráticas dentro
#> de cada cluster. Mede a coerência interna. 
#> Quanto menor, mais compactos os clusters.

#> Total Sum of Squares (TSS): Soma total das distâncias quadráticas
#> de todos os pontos ao centróide global
#> (a média de todas as observações). 
#> Mede a variabilidade total dos dados.

#> Proporção WSS/TSS: Mede quanto da variabilidade total ainda
#> permanece dentro dos clusters. 
#> Quanto menor, mais bem ajustado o agrupamento.

# 5) Tuning de k ---------------------------------------------------------------
res <- tune_cluster(
  wf,
  resamples = folds,
  grid      = grade,
  metrics   = mets,
  control   = control_grid(save_pred = TRUE, extract = identity)
)

autoplot(res)
tab  <- collect_metrics(res)
tab


sse_tbl <- tab |> 
  filter(.metric == "sse_ratio") |> 
  arrange(num_clusters)
sse_tbl

sse_tbl |>
  transmute(k = num_clusters,
            mean = round(mean, 4),
            se   = round(std_err, 4)) |>
  print()

# 6) Escolha de k pela regra do 1-SE ------------------------------------------
# Escolha o modelo mais simples cujo desempenho médio esteja  
# dentro de 1 erro padrão do melhor desempenho observado.

min_row <- sse_tbl |> slice_min(mean,
                                n = 1)

min_row

cutoff  <- min_row$mean + min_row$std_err
cutoff


best_k  <- sse_tbl |>
  filter(mean <= cutoff) |>
  slice_min(num_clusters, n = 1) |>
  pull(num_clusters)

best_k = 5

# 7) Ajuste final com k fixo ---------------------------------------------------
k_spec_final <- k_means(num_clusters = best_k) |>
  set_engine("stats", algorithm = "Hartigan-Wong", nstart = 50)

wf_final  <- workflow() |>
  add_recipe(rec) |>
  add_model(k_spec_final)

fit_final <- fit(wf_final, data = dados)


# 8) Predições, centros e matriz padronizada ----------------------------------
clusters <- predict(fit_final, new_data = dados) |>
  transmute(cluster = fct_drop(.pred_cluster))

km_engine <- extract_fit_engine(fit_final)    # objeto base::kmeans
centers_z <- km_engine$centers               # centros em z-score

# Centros em z-score no formato longo para heatmap
centros_long <- as_tibble(centers_z, rownames = "cluster") |>
  pivot_longer(-cluster, names_to = "variavel", values_to = "centro_z") |>
  mutate(cluster = factor(cluster)) |>
  arrange(cluster, variavel)

# Recuperar parâmetros da normalização e trazer centros para a escala original
norm_idx <- which(vapply(prep_rec$steps, 
                         inherits,
                         logical(1),
                         what = "step_normalize"))


norm_params <- tidy(prep_rec, 
                    number = norm_idx) |>
               select(terms, statistic, value) |>
               pivot_wider(names_from = statistic,
                           values_from = value)  

# Garantir mesma ordem das colunas
ord <- colnames(centers_z)
mp  <- match(ord, norm_params$terms)
mu  <- norm_params$mean[mp]
sg  <- norm_params$sd[mp]

# Desfazer padronização: x = z*sd + mean
centers_orig <- sweep(centers_z, 2, sg, `*`)
centers_orig <- sweep(centers_orig, 2, mu, `+`)
centers_orig <- as_tibble(centers_orig, rownames = "cluster") |>
  mutate(cluster = factor(cluster))



centers_orig |>
  relocate(cluster) |>
  mutate(across(-cluster, ~round(.x, 3))) |>
  print()

# Matriz padronizada para distâncias, silhueta e PCA
X_norm <- bake(prep_rec,
               new_data = dados) |> 
               as.matrix()

# 9) Silhueta ------------------------------------------------------------------
silhouette_df <- function(labels_factor, Xmat){
  lab_int <- as.integer(labels_factor)  # códigos 1..k
  sil     <- cluster::silhouette(lab_int, dist(Xmat))
  tibble(
    id        = seq_len(nrow(sil)),
    cluster   = sil[, "cluster"],
    neighbor  = sil[, "neighbor"],
    sil_width = sil[, "sil_width"]
  )
}
sil_tbl  <- silhouette_df(clusters$cluster, X_norm)
mean_sil <- mean(sil_tbl$sil_width)



# 10) Visualizações ------------------------------------------------------------
base_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "right",
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(size = 11),
    axis.title.x     = element_text(margin = margin(t = 6)),
    axis.title.y     = element_text(margin = margin(r = 6))
  )

# A) Curva tipo cotovelo
p_elbow <- sse_tbl |>
  ggplot(aes(x = num_clusters, y = mean)) +
  geom_ribbon(aes(ymin = mean - std_err, ymax = mean + std_err), alpha = 0.12) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_vline(xintercept = best_k, linetype = 2) +
  labs(
    title    = "Curva tipo cotovelo — sse_ratio (CV 5-fold)",
    subtitle = paste0("k escolhido pela regra do 1-SE: k = ", best_k),
    x        = "Número de clusters (k)",
    y        = "sse_ratio médio  (menor é melhor)"
  ) +
  base_theme

# B) Silhueta por cluster
p_sil <- sil_tbl |>
  mutate(cluster = factor(cluster)) |>
  ggplot(aes(x = cluster, y = sil_width)) +
  geom_boxplot(outlier.alpha = 0.15, width = 0.6) +
  geom_hline(yintercept = 0, linewidth = 0.6, linetype = 2) +
  annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.6,
           label = paste0("Silhueta média = ", number(mean_sil, accuracy = 0.001))) +
  labs(
    title = "Qualidade da partição — silhueta",
    x = "Cluster",
    y = "Largura de silhueta"
  ) +
  base_theme

# C) PCA bidimensional
pca   <- prcomp(X_norm, scale. = FALSE, center = FALSE)  # já normalizado no recipe
pc_df <- as_tibble(pca$x[, 1:2], .name_repair = ~c("PC1","PC2")) |>
  bind_cols(cluster = clusters$cluster)
var_exp <- summary(pca)$importance[2, 1:2]

p_pca <- ggplot(pc_df, aes(PC1, PC2, color = cluster)) +
  stat_ellipse(type = "norm", linewidth = 0.6, alpha = 0.3) +
  geom_point(alpha = 0.7, size = 1.7) +
  labs(
    title    = "Visualização em PCA",
    subtitle = paste0("Variância explicada: PC1 = ", percent(var_exp[1]),
                      ", PC2 = ", percent(var_exp[2])),
    x = "PC1",
    y = "PC2",
    color = "Cluster"
  ) +
  base_theme

# D) Heatmap de centros padronizados
p_centers <- centros_long |>
  ggplot(aes(x = variavel, y = cluster, fill = centro_z)) +
  geom_tile() +
  scale_fill_gradient2(low = muted("blue"), mid = "white", high = muted("red")) +
  labs(
    title = "Centros padronizados por variável",
    x     = "Variável",
    y     = "Cluster",
    fill  = "Centro (z-score)"
  ) +
  base_theme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Layout final
print((p_elbow | p_sil) / (p_pca | p_centers))




aa <- cbind(clusters, dados$Local)
  

ab <- aa %>%
  group_by(cluster) %>%
  summarise(locais = paste(unique(`dados$Local`), collapse = ", ")) %>%
  arrange(cluster)




#==========/==========/==========/==========/==========/==========/==========/==========/


dados2 <- rbind(clusters, dados_ml$Produtividade)








