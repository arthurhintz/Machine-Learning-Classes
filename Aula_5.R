# ======================================================================
# Projeto: Floresta Aleatória (Random Forest) com tidymodels (ISLR::Auto)
# Tópico:  pipeline profissional com recipes + workflow + tuning + VIP
# Autor :  prof. Thiago A. N. de Andrade
# Curso:   Machine Learning
# Data:    18/09/2025
# ======================================================================

# Pacotes ---------------------------------------------------------------
suppressMessages({
  library(tidymodels)  # recipes, rsample, parsnip, workflows, tune, yardstick, dials
  library(ISLR)        # dados Auto
  library(skimr)       # EDA
  library(GGally)      # ggpairs
  library(vip)         # variable importance
  library(doParallel)  # paralelismo no tuning
})


theme_set(ggplot2::theme_gray())
tidymodels_prefer()

# Reprodutibilidade -----------------------------------------------------
set.seed(8199510)

# 2) Dados ---------------------------------------------
# - Removemos 'name'
# - Mantemos 'mpg' na escala original. Discutimos no final, ok?
dados_raw <- ISLR::Auto |>
  tibble::as_tibble() |>
  dplyr::select(-name)

# 3) EDA objetiva (não interativa) --------------------------------------
dplyr::glimpse(dados_raw)
skimr::skim(dados_raw)

# Pares (amostra para performance)
set.seed(8199510)
amostra_pairs <- dplyr::sample_n(dados_raw, size = min(300, nrow(dados_raw)))
GGally::ggpairs(amostra_pairs)

# 4) Quebra treino/teste com estratificação aproximada ------------------
# (Estratificar por 'mpg' ajuda a manter distribuição semelhante)
set.seed(8199510)
split  <- rsample::initial_split(dados_raw, prop = 0.80, strata = mpg)
train  <- rsample::training(split)
test   <- rsample::testing(split)

# 5) Recipe (transformações reprodutíveis) ------------------------------
# - Fatores explícitos: cylinders, origin, year
# - Log nos preditores (não no alvo) com base-e
# - Tratamento de NA: mediana (numéricos) / moda (fatores)
# - step_novel para níveis inéditos no teste
rec <- recipes::recipe(
  mpg ~ displacement + horsepower + weight + cylinders + origin + year,
  data = train
) |>
  recipes::step_mutate(
    cylinders = factor(cylinders),
    origin    = factor(origin),
    year      = factor(year)
  ) |>
  recipes::step_log(displacement, horsepower, weight, base = exp(1)) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |>
  recipes::step_impute_median(recipes::all_numeric_predictors()) |>
  recipes::step_impute_mode(recipes::all_nominal_predictors())


rec_prep <- recipes::prep(rec)
p <- ncol(recipes::juice(rec_prep)) - 1L  
p

# 6) Especificação do modelo: Random Forest -----------------------------
rf_spec <- parsnip::rand_forest(
  mtry  = dials::tune(),
  trees = dials::tune(),
  min_n = dials::tune()
) |>
  parsnip::set_engine(
    "ranger",
    importance = "permutation"  
  ) |>
  parsnip::set_mode("regression")

# 7) Workflow (modelo + recipe) -----------------------------------------
wf <- workflows::workflow() |>
  workflows::add_model(rf_spec) |>
  workflows::add_recipe(rec)

# 8) Reamostragem (CV) --------------------------------------------------
set.seed(8199510)
folds <- rsample::vfold_cv(train, v = 5, strata = mpg)

# 9) Espaço de hiperparâmetros ------------------------------------------

grid <- dials::grid_latin_hypercube(
  dials::trees(range = c(300L, 1500L)),
  dials::mtry(range = c(1L, p)),
  dials::min_n(range = c(2L, 25L)),
  size = 100
)

# 10) Paralelismo -------------------------------------------------------
n_cores <- max(1L, parallel::detectCores() - 1L)
doParallel::registerDoParallel(cores = n_cores)

ctrl <- tune::control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  parallel_over = "resamples",
  verbose       = FALSE
)

# 11) Tuning ------------------------------------------------------------
set.seed(8199510)
res <- tune::tune_grid(
  object     = wf,
  resamples  = folds,
  grid       = grid,
  metrics    = yardstick::metric_set(
    yardstick::rmse, yardstick::rsq, yardstick::mae, yardstick::mape
  ),
  control    = ctrl
)

# 12) Diagnóstico do tuning --------------------------------------------
autoplot(res)  
tune::show_best(res, metric = "rmse", n = 10)

# Seleção: melhor RMSE
best_rmse <- tune::select_best(res, metric = "rmse")
best_rmse

# 13) Finalização e avaliação em teste ---------------------------------
final_wf  <- tune::finalize_workflow(wf, best_rmse)

# last_fit: ajusta no treino completo e avalia no teste do split
final_fit <- tune::last_fit(final_wf, split)

# Métricas no teste
metrics_test <- tune::collect_metrics(final_fit)
metrics_test

# Predições no teste 
pred_test <- tune::collect_predictions(final_fit)
pred_test |>
  yardstick::metrics(truth = mpg, estimate = .pred)  # rmse, rsq etc.

# 14) Importância de variáveis  ------------------------
wf_trained  <- workflows::extract_workflow(final_fit)
fit_parsnip <- workflows::pull_workflow_fit(wf_trained)  


vip::vip(fit_parsnip$fit, num_features = 10)

# Erro OOB da floresta (regressão: MSE; RMSE = sqrt(MSE))
oob_mse  <- fit_parsnip$fit$prediction.error
oob_rmse <- sqrt(oob_mse)
oob_rmse

# 15) Opcional: salvar objetos ------------------------------------------
# saveRDS(final_wf,  file = "final_workflow_rf.rds")
# saveRDS(final_fit, file = "final_fit_rf_lastfit.rds")

# 16) Encerrar paralelismo ----------------------------------------------
doParallel::stopImplicitCluster()
