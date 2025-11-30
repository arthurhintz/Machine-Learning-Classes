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
  library(lubridate)
  })

#library(recipes)
#library(parsnip)
tidymodels::tidymodels_prefer()

theme_set(ggplot2::theme_grey(base_size = 13))

set.seed(1248)

dados_ml <- read.csv2("dados_ml.csv") 

dados_ml$dias <- yday(dados_ml$Epoca_de_semeadura)

dados <- dados_ml %>% 
  select(-c(COD_PROD, Cod_Estacao_Met, Area_colhida, Epoca_de_semeadura, Cultivar, Produtividade)) %>%
  drop_na() |> 
  dplyr::mutate(across(where(is.numeric), as.numeric),
                across(!where(is.numeric), as.factor))


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

tab$.metric

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

best_k = 4

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


# C) PCA bidimensional
pca   <- prcomp(X_norm, scale. = FALSE, center = FALSE)  # já normalizado no recipe
pc_df <- as_tibble(pca$x[, 1:2], .name_repair = ~c("PC1","PC2")) |>
  bind_cols(cluster = clusters$cluster)
var_exp <- summary(pca)$importance[2, 1:2]

p_pca <- ggplot(pc_df, aes(PC1, PC2, color = cluster)) +
  stat_ellipse(type = "norm", linewidth = 0.6, alpha = 0.3) +
  geom_point(alpha = 0.7, size = 1.7) +
  labs(
    title    = "",
    subtitle = paste0("Variância explicada: PC1 = ", percent(var_exp[1]),
                      ", PC2 = ", percent(var_exp[2])),
    x = "PC1",
    y = "PC2",
    color = "Cluster"
  ) +
  base_theme

p_pca
#ggsave("pca.pdf", p_pca, width = 5, height = 3, 
 #      units = "in", dpi = 200)



aa <- cbind(clusters, dados$Local)
  

ab <- aa %>%
  group_by(cluster) %>%
  summarise(locais = paste(unique(`dados$Local`), collapse = ", ")) %>%
  arrange(cluster)

View(ab)


#==========/==========/==========/==========/==========/==========/==========/==========/


dados2 <- dados_ml %>% 
  select(-c(COD_PROD, Cod_Estacao_Met, Area_colhida, Epoca_de_semeadura, Cultivar)) %>%
  drop_na() |> 
  dplyr::mutate(across(where(is.numeric), as.numeric),
                across(!where(is.numeric), as.factor))

dados_f <- cbind(clusters, dados2$Produtividade)

colnames(dados_f) <- c("cluster", "prod")


# Treino/teste ####
base_quebra <- rsample::initial_split(dados_f)
treino <- rsample::training(base_quebra)
teste  <- rsample::testing(base_quebra)

# Especificação do modelo: Árvore de Regressão ####
modelo <- parsnip::decision_tree(
  cost_complexity = tune::tune(),
  tree_depth      = tune::tune(),
  min_n           = tune::tune()
) |>
  parsnip::set_engine("rpart") |>
  parsnip::set_mode("regression")

# Validação cruzada ####

bases_Cross <- rsample::vfold_cv(treino, v = 5)

# Grade de hiperparâmetros ####
grade <- dials::grid_latin_hypercube(
  dials::cost_complexity(),
  dials::tree_depth(),
  dials::min_n(),
  size = 300
)

# Tunagem ####
tunagem <- tune::tune_grid(
  object       = modelo,
  preprocessor = prod ~ cluster,
  resamples    = bases_Cross,
  grid         = grade,
  metrics      = yardstick::metric_set(
    yardstick::rmse,
    yardstick::rsq,
    yardstick::mae,
    yardstick::mape
  ),
  control      = tune::control_grid(
    verbose   = TRUE,
    allow_par = FALSE
  )
)

# Resultados da tunagem ####
autoplot(tunagem)

tune::show_best(tunagem, metric = "rsq", n = 10)

melhores <- tune::select_best(tunagem, metric = "rmse")
melhores

# Finalização e ajuste no treino ####
modelo_final <- tune::finalize_model(modelo, melhores)

ajuste_final <- parsnip::fit(
  object  = modelo_final,
  formula = prod ~ cluster,
  data    = treino
)

# Predição no teste ####
pred_teste <- stats::predict(ajuste_final, new_data = teste)

# Métricas no teste ####
yardstick::rmse(dplyr::tibble(truth = teste$prod, estimate = pred_teste$.pred), truth, estimate)
yardstick::mae (dplyr::tibble(truth = teste$prod, estimate = pred_teste$.pred), truth, estimate)
yardstick::rsq (dplyr::tibble(truth = teste$prod, estimate = pred_teste$.pred), truth, estimate)

# Importância de variáveis ####
vip::vip(ajuste_final)



bb <- cbind(teste, pred_teste)


ab <- bb %>%
  group_by(cluster) %>%
  summarise(producao = mean(.pred)) 

View(ab)

#==========/==========/==========/==========/==========/==========/==========/==========/

#==========/==========/==========/==========/==========/==========/==========/==========/
# Mapa com os clusters

library(geobr)
library(sf)

#prod_c <- stats::predict(ajuste_final, new_data = dados_f)



produtividade <- dados2 |> 
  group_by(Local) |> 
  summarise(prod = mean(Produtividade))


rs <- read_state(code_state = "RS", year = 2020)

bd <- data.frame(
  Latitude = dados$Latitude,
  Longitude = dados$Longitude,
  Cluster = clusters,
  Prod = dados2$Produtividade
)



  



bd_sf <- st_as_sf(bd, coords = c("Longitude", "Latitude"), crs = 4326)


# 3. Plotar
graph1 <- ggplot() +
  geom_sf(data = rs, fill = "gray95", color = "gray60") +
  geom_sf(data = bd_sf, aes(color = cluster, size = Prod), alpha = 0.8) +
  scale_size_continuous(range = c(2, 8)) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(
    title = "Agrupamento",
    color = "Grupo",
    size = "Produtividade"
  )

graph1

ggsave("mapa_cluster.pdf", graph1, width = 5, height = 3, 
       units = "in", dpi = 200)


# opção 2


# Transformar para projeção métrica (em metros)
bd_proj <- st_transform(bd_sf, 31983)  # SIRGAS / RS

# Criar buffer de 30 km (30000 metros)
raio_km <- 100000
buf <- st_buffer(bd_proj, dist = raio_km)

# Dissolver buffers por cluster (cria áreas contínuas)
regioes <- buf %>%
  group_by(cluster) %>%
  summarise(geometry = st_union(geometry)) %>%
  st_as_sf()

# Voltar para WGS84 para plotar
regioes <- st_transform(regioes, 4326)

# Plot
ggplot() +
  geom_sf(data = rs, fill = "gray95", color = "gray70") +
  geom_sf(data = regioes, aes(fill = cluster), alpha = 0.45, color = NA) +
  geom_sf(data = bd_sf, aes(color = cluster, size = Prod), alpha = 0.9) +
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  scale_size_continuous(range = c(2, 8)) +
  theme_minimal() +
  labs(
    title = "Regiões por Cluster (buffer de 30 km)",
    fill = "Cluster",
    color = "Cluster",
    size = "Produtividade"
  )


# opção 3

# 3. Gerar Voronoi no bounding box do RS
vor <- st_voronoi(st_union(bd_sf), envelope = st_geometry(rs))

# 4. Converter para sf
vor_sf <- st_collection_extract(vor)

# 5. Atribuir cluster a cada polígono (por localização)
vor_df <- st_sf(geometry = vor_sf) |>
  st_join(bd_sf, join = st_contains)

rs <- st_transform(rs, 4326)

# 6. Recortar (clip) os polígonos pelo RS
vor_clip <- st_intersection(vor_df, rs)

# 7. Plotar
ggplot() +
  geom_sf(data = rs, fill = "gray95", color = "gray70") +
  geom_sf(data = vor_clip, aes(fill = cluster), alpha = 0.5, color = NA) +
  geom_sf(data = bd_sf, aes(color = cluster, size = Prod), alpha = 0.9) +
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(
    title = "Regiões por Cluster (Voronoi Recortado pelo RS)",
    fill = "Cluster",
    color = "Cluster",
    size = "Prod"
  )




# opção 4


# 1. Mapa RS
rs <- read_state(code_state = "RS", year = 2020)
rs <- st_transform(rs, 31983)  # projeta para metros

prod_c

round(prod_c$.pred, 2)

# 2. Dados como sf
bd_sf <- st_as_sf(
  data.frame(
    Latitude = dados$Latitude,
    Longitude = dados$Longitude,
    Cluster = clusters,
    Prod = factor(round(prod_c$.pred, 2))
  ),
  coords = c("Longitude", "Latitude"),
  crs = 4326
) |> st_transform(31983)


# 3. Criar buffer de 50 km (50.000 metros)
buffer_50km <- st_buffer(bd_sf, dist = 100000)

# 4. Unir buffers
area_valida <- st_union(buffer_50km)

# 5. Criar Voronoi limitado pelo RS
vor <- st_voronoi(st_union(bd_sf), envelope = st_geometry(rs))
vor_sf <- st_collection_extract(vor) |> st_sf()

# 6. Atribuir cluster aos polígonos
vor_df <- st_join(vor_sf, bd_sf, join = st_nearest_feature)

# 7. Recortar: Voronoi ∩ RS ∩ buffer
vor_clip <- st_intersection(vor_df, rs) |> 
  st_intersection(area_valida)

# 8. Voltar o RS para WGS84 só para plot
rs_plot <- st_transform(rs, 4326)
bd_plot <- st_transform(bd_sf, 4326)
vor_plot <- st_transform(vor_clip, 4326)


# 9. Plot
graph2 <- ggplot() +
  geom_sf(data = rs_plot, fill = "gray95", color = "gray70") +
  geom_sf(data = vor_plot, aes(fill = Prod), alpha = 0.6, color = NA) +
  geom_sf(data = bd_plot, aes(color = Prod), alpha = 0.8) +
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(
    title = "",
    fill = "Cluster",
    color = "Cluster",
    size = "Prod"
  )

graph2
ggsave("mapa_prod.pdf", graph2, width = 5, height = 3, 
       units = "in", dpi = 200)




