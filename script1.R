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
  select(-c(COD_PROD, Cod_Estacao_Met, Area_colhida, Epoca_de_semeadura, Cultivar, Produtividade)) %>%
  drop_na() |>dplyr::mutate(across(where(is.numeric), as.numeric),
                across(!where(is.numeric), as.factor))

# summary(dados$Produtividade)
View(teste)



#==========/==========/==========/==========/==========/==========/==========/==========/
# corplot pdf


dados <- na.omit(dados)

numeric_data <- dados %>% 
  select_if(is.numeric)

corr_matrix <- cor(numeric_data)


pdf("corplot.pdf", width = 6, height = 6)  
par(mfrow = c(1,1))
par(mar = c(1, 1, 1, 1))    
corrplot(
  corr_matrix,
  method = "circle",
  type = "lower",
  order = "hclust",
  tl.cex = 0.7,              # tamanho dos rótulos das variáveis
  cl.cex = 0.7,              # tamanho da legenda (barra de cores)
  diag = FALSE               # opcional — remove diagonal
)

dev.off()