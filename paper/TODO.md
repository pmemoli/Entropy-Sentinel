- Contextualizar con el campo de "Predicting Out-of-Distribution Performance" y papers relevantes post 2022 (ATC).

- Que el baseline sea ATC, no lo que hice.

- Mencionar que lo novedoso es que lo hacemos sobre LLMs que hasta donde sabemos no se hace, con excepcion de "Predicting the Performance of Foundation Models via Agreement-on-the-Line", pero que hace ensembles de modelos y no es apto para monitoring.

- Agregar discusion de como puede utilizarse para monitoring e improvement, y que es lo que demostramos en concreto en este research.

- Mas mechanistic analysis.

Tomorrow leo un overview de esto y pienso experimentos concretos:

- Introducir reward models como la solucion estandar para improvement. Que problemas tienen? 
- Introducir LLM as a judge como la solucion estandar para monitoring. Que problemas tienen?

- Algun experimento de monitoreo concreto comparando con reward models.

TODO:

- Review reward models.
- Correr WildBench y MT-Bench para Phi 3.5 (misma configuracion).
- Correr el judge.
- Entrenar sentinel y evaluar...
