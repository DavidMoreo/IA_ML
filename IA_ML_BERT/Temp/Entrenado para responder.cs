//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using Microsoft.ML;
//using Microsoft.ML.Data;
//using Newtonsoft.Json;

//class Program
//{
//    static void Main(string[] args)
//    {
//        // Crear el contexto de ML
//        var context = new MLContext();

//        // Leer los datos desde el archivo JSON y deserializarlos
//        var jsonData = File.ReadAllText("qaData.json");
//        var qaDataList = JsonConvert.DeserializeObject<List<QAData>>(jsonData);

//        // Convertir la lista a IDataView
//        var data = context.Data.LoadFromEnumerable(qaDataList);

//        // Crear el pipeline de entrenamiento
//        var pipeline = context.Transforms.Conversion.MapValueToKey("Answer")
//            .Append(context.Transforms.Text.FeaturizeText("Question", "Question"))
//            .Append(context.Transforms.Concatenate("Features", "Question"))
//            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Answer", "Features"))
//            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

//        // Entrenar el modelo
//        var model = pipeline.Fit(data);

//        // Hacer predicciones
//        var predictions = model.Transform(data);

//        // Evaluar el modelo
//        var metrics = context.MulticlassClassification.Evaluate(predictions, "Answer");
//        Console.WriteLine($"Log-loss: {metrics.LogLoss}");

//        // Guardar el modelo
//        context.Model.Save(model, data.Schema, "qaModel.zip");

//        Console.WriteLine("Modelo entrenado y guardado.");

//        // Cargar el modelo guardado
//        ITransformer loadedModel = context.Model.Load("qaModel.zip", out var modelInputSchema);

//        // Crear una instancia del predictor
//        var predictor = context.Model.CreatePredictionEngine<QAData, QADataPrediction>(loadedModel);

//        // Solicitar una pregunta al usuario
//        Console.WriteLine("Ingrese una pregunta:");
//        string userQuestion = Console.ReadLine();

//        // Realizar la predicción
//        var prediction = predictor.Predict(new QAData { Question = userQuestion });

//        // Mostrar la respuesta
//        Console.WriteLine($"Respuesta: {prediction.Answer}");
//    }
//}

//// Clases de datos
//public class QAData
//{
//    public string Question { get; set; }
//    public string Answer { get; set; }
//}

//public class QADataPrediction
//{
//    [ColumnName("PredictedLabel")]
//    public string Answer { get; set; }
//}
