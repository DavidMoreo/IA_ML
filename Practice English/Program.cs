using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

public class InputData
{
    public string Context { get; set; }
    public string Question { get; set; }
    public string Fragment { get; set; } // Fragmento del contexto que contiene la respuesta
}

public class OutputPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedFragment { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        // Crear el contexto de ML.NET
        var mlContext = new MLContext();

        // Datos de ejemplo para el entrenamiento (Contexto largo + Pregunta + Fragmento de respuesta)
        var trainingData = new List<InputData>
        {
            new InputData
            {
                Context = "Microsoft es una empresa multinacional de tecnología que produce software, hardware, y servicios en la nube. Microsoft fue fundada por Bill Gates y Paul Allen. La sede principal está en Redmond, Washington.",
                Question = "¿Quién fundó Microsoft?",
                Fragment = "Microsoft fue fundada por Bill Gates y Paul Allen."
            },
            new InputData
            {
                Context = "Microsoft es una empresa multinacional de tecnología que produce software, hardware, y servicios en la nube. Microsoft fue fundada por Bill Gates y Paul Allen. La sede principal está en Redmond, Washington.",
                Question = "¿Dónde está la sede de Microsoft?",
                Fragment = "La sede principal está en Redmond, Washington."
            },
            new InputData
            {
                Context = "Microsoft es una empresa multinacional de tecnología que produce software, hardware, y servicios en la nube. Microsoft fue fundada por Bill Gates y Paul Allen. La sede principal está en Redmond, Washington.",
                Question = "¿Qué produce Microsoft?",
                Fragment = "Microsoft produce software, hardware, y servicios en la nube."
            }
        };

        // Cargar los datos de entrenamiento
        var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

        // Crear el pipeline para transformar el texto y entrenar el modelo
        var pipeline = mlContext.Transforms.Text.FeaturizeText("ContextFeaturized", nameof(InputData.Context))
            .Append(mlContext.Transforms.Text.FeaturizeText("QuestionFeaturized", nameof(InputData.Question)))
            .Append(mlContext.Transforms.Concatenate("Features", "ContextFeaturized", "QuestionFeaturized"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(InputData.Fragment))) // Usamos el fragmento como label
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

        // Entrenar el modelo
        var model = pipeline.Fit(trainingDataView);

        // Crear datos de prueba (Contexto completo + Pregunta)
        var testData = new List<InputData>
        {
            new InputData
            {
                Context = "Microsoft es una empresa multinacional de tecnología que produce software, hardware, y servicios en la nube. Microsoft fue fundada por Bill Gates y Paul Allen. La sede principal está en Redmond, Washington.",
                Question = "¿Qué produce Microsoft?"
            }
        };

        // Cargar los datos de prueba
        var testDataView = mlContext.Data.LoadFromEnumerable(testData);

        // Hacer predicción con el modelo entrenado
        var predictions = model.Transform(testDataView);
        var predictedResults = mlContext.Data.CreateEnumerable<OutputPrediction>(predictions, reuseRowObject: false);

        // Mostrar los resultados de la predicción (fragmento relevante)
        foreach (var prediction in predictedResults)
        {
            Console.WriteLine($"Fragmento relevante: {prediction.PredictedFragment}");
        }
    }
}
