//// Crear un esquema de datos con una longitud fija de 8 para la entrada
//var data = new List<InputData>();

//var dataView = context.Data.LoadFromEnumerable(data);

//var pipeline = context.Transforms.ApplyOnnxModel(
//    modelFile: modelPath,
//    inputColumnNames: new[] { "input" },
//    outputColumnNames: new[] { "output" }
//);

//var model = pipeline.Fit(dataView);

//var predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(model);

//// Asegúrate de que los valores de entrada tengan una longitud fija de 8
//var input = new InputData { Input = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } }; // Reemplaza con los valores reales
//var prediction = predictionEngine.Predict(input);

//Console.WriteLine($"Respuesta generada: '{string.Join(", ", prediction.Output)}'");