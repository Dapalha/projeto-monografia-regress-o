# Exemplos

Duas planilhas de auditoria reais, anonimizadas, usadas para validar o sistema:

- **69251992613cd7d158638cb9_auditoria.csv** — rastreador com fraude confirmada. O arquivo
  registra oito episódios de violação de alimentação: a tensão externa cai de cerca de 14 V
  para 3 a 4 V enquanto o veículo se desloca a mais de 80 km/h com a ignição desligada, e os
  pontos chegam em bloco como ROUTE'S RECONSTRUCTION (o rastreador estava na bateria de
  backup e transmitiu depois). Em cada episódio aparecem também POWER OFF e DEVICE POWER ON.

- **65c4c75f8cc6500f47d36bc1_auditoria.csv** — rastreador sem fraude. Tem 58 episódios de
  interferência de sinal registrados como jammer, todos com duração entre 1 e 5 minutos,
  nenhum chegando ao limite de 30 minutos que caracteriza fraude.

## Formato

Planilhas exportadas da plataforma de auditoria, em CSV com separador ";" e as colunas:
Location date, Creation date, Delay, Reception date, Reception delay, Reason Code, Speed,
Ignition, GPS Satellites, GSM Signal, HDOP, HDOP STATUS, External Voltage, Internal Battery,
Hodometer, Block Car, Lat, Lng, Ibutton, Module Id, Plate e Place.

## Como usar

Na seção 10 do notebook, envie o arquivo pelo computador ou copie-o para a pasta
Meu Drive/dados_brutos e ajuste a variável ARQUIVO.
