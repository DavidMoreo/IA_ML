using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

public class TextData
{
    public string Text { get; set; }
    public string Label { get; set; } // Agregar una columna de etiquetas
}

public class NextWordPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedWord { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Cargar datos
        var data = new List<TextData>
        {
            new TextData { Text = "hola mundo", Label = "mundo" },
            new TextData { Text = "hola amigo", Label = "amigo" },
            new TextData { Text = "buenos dias", Label = "dias" },
            new TextData { Text = "buenas noches", Label = "noches" },
            // Agrega más datos aquí
        };

        var trainingData = mlContext.Data.LoadFromEnumerable(data);

        // Preprocesar datos
        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(TextData.Label))); // Convertir etiquetas a claves

        // Definir el algoritmo de aprendizaje
        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

        // Crear el modelo de entrenamiento
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        // Entrenar el modelo
        Console.WriteLine("Entrenando el modelo...");
        var model = trainingPipeline.Fit(trainingData);
        Console.WriteLine("Modelo entrenado.");

        // Hacer predicciones
        var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, NextWordPrediction>(model);

        var input = new TextData { Text = "hola" };
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"La siguiente palabra podría ser: {prediction.PredictedWord}");
    }
}