# TFG- Helena García Osorio
Código fonte para a realización do Traballo de Fin de Grao.


# 1-Clasificador de Estrés BERT Híbrido

O script train_bert_vanila.py permite adestrar un modelo baseado en BERT-base-uncased para a clasificación binaria de presenza ou non presenza de estrés sobre o corpus Dreaddit. Durante o adestramento o modelo combina as representacións contextuais xeradas por BERT coas características emocionales obtidas mediante BoSE.

Para executar o adestramento é necesario dispor dun entorno con Python 3.10. e todas as dependencias do proxecto instaladas. No desenvolvemento deste TFG empregouse un entorno virtual denominado bert_env, non obstante, pode emptregarse calquiera entorno que contenga as bibliotecas precisas.

## Datos de entrada:
O script requiere os seguintes arquivos de entrada:
- dreaddit-train.csv: conxunto de adestramento, debe conter al menos as columnas text e label.
- dreaddit-test.csv: conxunto de proba coa mesma estrutura.
- bose_train.npy: representacións BoSE correspondentes a cada publicación do conxunto de adestramento
- bose_test.npy: representacións BoSE correspondentes ao conxunto de proba

É importante cos arquivos .npy manteñan exactamente o mesmo orden e número de instancias cos ficheiros CSV.

Para executar o adestramento no CESGA empregase o escript run_vanila.sh, que solicita unha GPU, 4 CPU, 16GB de memoria RAM  e un tempo máximo de dúas horas. O script carga o entorno Python correspondente e executa automáticamente o adestramento, indicando as rutas dos datos de entrada e o directorio onde se almancenarán os resultados.

# 2-Transferencia de Coñecemento

O Script tranferencia_depresion_bert.py permite aplicar o modelo BERT + BoSE adestrado previamente sobre o corpus Dreaddit ao corpus eRisk.

Antes de executar o srcipt é necesario dispoñer do modelo BERT + BoSE previamente adestrado así como de todas as dependencias do proxecto instaladas. O modelo debe conter o ficheiro best_model.pt xerado durante a fase de adestramento.

## Datos de entrada:
O script require dos seguintes arquivos:
- df_nivelPosts_etiquetados.csv: conxunto de publicacións do corpus eRisk.
- bose_depression_sparse.npz: representacións BoSE de todas as publicacións do corpus eRisk.
- best_model.pt: modelo BERT + BoSE previamente adestrado sobre o dominio de estrés.

É importante que o arquivo bose_depression_sparse.npz manteña exactamente a mesma orde e número de publicacións co ficheiro df_nivelPosts_etiquetado.csv

Para executar a inferenza no CESGA pode empregarse run_depresion.sh, que solicita unha GPU, 4 CPU, 16GB de memoria RAM e un tempo máximo de execución de dúas horas. O script carga o entorno de Python correspondente e executa automáticamente a inferencia indicando as rutas do conxunto de datos, as características BoSE e o directorio onde se almacenan os resultados.


