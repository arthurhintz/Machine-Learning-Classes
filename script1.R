library(tidyverse)

dados <- read.csv2("dados_ml.csv") 
  
dados$dias <- yday(dados$Epoca_de_semeadura)

teste <- dados %>% 
  mutate(
    prod = case_when(
      Produtividade <= quantile(Produtividade, 0.25)  ~ "Baixa",
      Produtividade <= quantile(Produtividade, 0.5)  ~ "Media",
      Produtividade <= quantile(Produtividade, 0.75)  ~ "Alta",
      TRUE      ~ "Altissima"
    )
  ) %>% 
  select(-c(COD_PROD, Cod_Estacao_Met, Area_colhida, Epoca_de_semeadura, Cultivar)) %>%
  drop_na()

# summary(dados$Produtividade)
View(teste)
