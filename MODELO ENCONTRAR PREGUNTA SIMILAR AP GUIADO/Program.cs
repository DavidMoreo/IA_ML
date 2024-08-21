using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

public class QuestionPair
{
    public string Question { get; set; }
    public string Answer { get; set; }
    public bool Label { get; set; }
}

public class QuestionPairInput
{
    [LoadColumn(0)]
    public string Question { get; set; }

    [LoadColumn(1)]
    public string Answer { get; set; }

    [LoadColumn(2)]
    public bool Label { get; set; }
}

public class QuestionPairOutput
{
    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
}

public class QuestionAnswerModel
{
    // Método para cargar datos desde archivos JSON en una carpeta
    public static List<QuestionPair> LoadDataFromFolder(string folderPath)
    {
        var trainingData = new List<QuestionPair>();

        foreach (var filePath in Directory.GetFiles(folderPath, "*.json"))
        {
            var jsonData = File.ReadAllText(filePath);
            var questionPairs = JsonConvert.DeserializeObject<List<QuestionPair>>(jsonData);

            // Filtrar las palabras vacías de las preguntas
            foreach (var pair in questionPairs)
            {
                pair.Question = string.Join(" ", FilterStopWords(pair.Question.Split(' ')));
            }

            trainingData.AddRange(questionPairs);
        }

        return trainingData;
    }

    // Método para entrenar el modelo
    public static void TrainModel(string folderPath)
    {
        // Cargar los datos
        var trainingData = LoadDataFromFolder(folderPath);

        // Crear el contexto de ML.NET
        var mlContext = new MLContext();

        // Convertir los datos en un IDataView
        var data = mlContext.Data.LoadFromEnumerable(trainingData);

        // Preprocesamiento de datos (Convertir texto en números)
        var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(QuestionPair.Question))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(QuestionPair.Label)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        // Algoritmo de clasificación binaria
        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: "Label",
            maximumNumberOfIterations: 100);

        // Entrenamiento del modelo
        var trainingPipeline = dataPipeline.Append(trainer);
        var model = trainingPipeline.Fit(data);

        // Guardar el modelo entrenado
        mlContext.Model.Save(model, data.Schema, "questionAnswerModel.zip");
    }

    // Método para predecir la respuesta basada en una pregunta
    public static bool PredictAnswer(string question)
    {
        var mlContext = new MLContext();

        // Filtrar palabras vacías de la pregunta
        var filteredQuestion = string.Join(" ", FilterStopWords(question.Split(' ')));

        // Cargar el modelo entrenado
        var model = mlContext.Model.Load("questionAnswerModel.zip", out var schema);

        // Crear un predictor
        var predictor = mlContext.Model.CreatePredictionEngine<QuestionPairInput, QuestionPairOutput>(model);

        // Predecir la respuesta
        var input = new QuestionPairInput { Question = filteredQuestion };
        var result = predictor.Predict(input);

        return result.PredictedLabel;
    }

    // Método para filtrar palabras vacías de un conjunto de tokens
    static string[] FilterStopWords(string[] tokens)
    {
        var stopwords = new HashSet<string>(new[]
        {
            "el", "la", "los", "las", "de", "en", "y", "a", "que", "es", "con", "por", "como", "para", "un", "una", "al", "se",
            "él", "ella", "ellos", "ellas", "del", "á", "é", "í", "ó", "ú", "áéíóú", "cómo", "cuándo", "dónde", "qué", "qué"
            // Añade más palabras comunes acentuadas si es necesario
        });

        var list = tokens.Where(token => !stopwords.Contains(token.ToLower())).ToArray();

        for (int i = 0; i < list.Length; i++)
        {
            list[i] = CleanStopword(list[i]);
        }

        return list;
    }

    // Método para limpiar una palabra si es necesario
    static string CleanStopword(string token)
    {
        // Aquí podrías añadir lógica adicional para limpiar el token si es necesario
        return token;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo (esto lo haces una vez)
        QuestionAnswerModel.TrainModel("trainingData/");

        // Bucle infinito para permitir preguntas continuas
        while (true)
        {
            Console.WriteLine("Ingrese su pregunta (o 'salir' para terminar):");
            var question = Console.ReadLine();

            if (string.Equals(question, "salir", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            // Usar el modelo para hacer una predicción solo con la pregunta
            bool answer = QuestionAnswerModel.PredictAnswer(question);

            Console.WriteLine($"Respuesta: '{(answer ? "Sí" : "No")}'");
        }
    }
}
