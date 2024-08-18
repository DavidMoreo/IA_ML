using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext();

        // Cargar datos de entrenamiento desde JSON
        var trainingData = LoadDataFromJson<QuestionPair>("trainingData.json");

        // Convertir las listas a IDataView
        var trainData = context.Data.LoadFromEnumerable(trainingData);

        // Crear el pipeline de entrenamiento
        var pipeline = context.Transforms.Text.FeaturizeText("Question1", nameof(QuestionPair.Question1))
            .Append(context.Transforms.Text.FeaturizeText("Question2", nameof(QuestionPair.Question2)))
            .Append(context.Transforms.Concatenate("Features", "Question1", "Question2"))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("SameIntention", "Features"));

        // Entrenar el modelo
        var model = pipeline.Fit(trainData);

        // Evaluar el modelo
        var predictions = model.Transform(trainData);
        var metrics = context.BinaryClassification.Evaluate(predictions, "SameIntention");

        Console.WriteLine($"Precisión del modelo: {metrics.Accuracy:P2}");
        Console.WriteLine($"Área bajo la curva ROC (AUC-ROC): {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"Log-loss: {metrics.LogLoss:P2}");

        // Guardar el modelo
        context.Model.Save(model, trainData.Schema, "qaModel.zip");
        Console.WriteLine("Modelo entrenado y guardado.");

        // Probar el modelo con una pregunta del cliente
        TestModel(context, model, 0.1f); // Ajustar el umbral de confianza según sea necesario
    }

    static IEnumerable<T> LoadDataFromJson<T>(string filePath)
    {
        var jsonData = File.ReadAllText(filePath);
        return JsonConvert.DeserializeObject<List<T>>(jsonData);
    }

    static void TestModel(MLContext context, ITransformer model, float confidenceThreshold)
    {
        var predictor = context.Model.CreatePredictionEngine<QuestionPair, QuestionPairPrediction>(model);

        // Lista de preguntas almacenadas (productos diferentes)
        var storedQuestions = new List<QuestionPair>
        {
            new QuestionPair { Question2 = "¿Como usar sensor de temperatura a un Arduino?" },
            new QuestionPair { Question2 = "¿Qué sensores se pueden conectar a un Arduino?" },
            new QuestionPair { Question2 = "¿Cómo se usa un servomotor con Arduino?" },
            new QuestionPair { Question2 = "¿Qué es una placa Arduino y para qué se usa?" },
            new QuestionPair { Question2 = "¿Cómo se realiza un proyecto de control de luces con Arduino?" }
        };

        // Pregunta del cliente para comparar
        var clientQuestion = "¿Cómo puedo comprar un sensor de temperatura a un Arduino?";

        // Evaluar y seleccionar el par de preguntas almacenadas que más se asemeje a la pregunta del cliente
        var bestMatch = storedQuestions
            .Select(storedQuestion => new
            {
                StoredQuestion = storedQuestion,
                Prediction = predictor.Predict(new QuestionPair
                {
                    Question1 = clientQuestion,
                    Question2 = storedQuestion.Question2 // Usamos Question2 del storedQuestion para la comparación
                })
            })
            .OrderByDescending(result => result.Prediction.Score)
            .FirstOrDefault(result => result.Prediction.Score >= confidenceThreshold); // Filtrar por umbral de confianza

        if (bestMatch != null)
        {
            Console.WriteLine("La pregunta almacenada más similar a la pregunta del cliente es:");
            Console.WriteLine($"Pregunta del cliente: {clientQuestion}");
            Console.WriteLine($"Pregunta almacenada: {bestMatch.StoredQuestion.Question2}");
            Console.WriteLine($"Probabilidad de coincidencia:{bestMatch.Prediction.Score}  {bestMatch.Prediction.Score:P2}");
            Console.WriteLine($"Intención: {(bestMatch.Prediction.PredictedLabel ? "Misma intención" : "Intenciones diferentes")}");
        }
        else
        {
            Console.WriteLine("No se encontró una coincidencia suficientemente confiable.");
        }
    }
}

// Clases de datos
public class QuestionPair
{
    public string Question1 { get; set; }
    public string Question2 { get; set; }
    public bool SameIntention { get; set; }
}

public class QuestionPairPrediction
{
    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
}
