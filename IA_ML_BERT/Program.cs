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
        //var trainingDataWarranty = LoadDataFromJson<QuestionPair>("trainingData/trainingDataWarranty.json");
        //var trainingDataInstructionsForUse = LoadDataFromJson<QuestionPair>("trainingData/trainingDataInstructionsForUse.json");

        // Concatenar los datos de los dos conjuntos
         var folderPath = "trainingData/";

        // Cargar y combinar datos desde todos los archivos JSON en la carpeta

        var trainingData = LoadDataFromFolder<QuestionPair>(folderPath);



        // Convertir las listas a IDataView
        var trainData = context.Data.LoadFromEnumerable(trainingData);

        // Crear el pipeline de entrenamiento
        var pipeline = context.Transforms.Text.TokenizeIntoWords("Tokens1", nameof(QuestionPair.Question1))
            .Append(context.Transforms.Text.TokenizeIntoWords("Tokens2", nameof(QuestionPair.Question2)))
            .Append(context.Transforms.CustomMapping<FilterStopWordsInput, FilterStopWordsOutput>(
                (input, output) =>
                {
                    output.Tokens1 = FilterStopWords(input.Tokens1);
                    output.Tokens2 = FilterStopWords(input.Tokens2);
                }, "FilterStopWords"))
            .Append(context.Transforms.Text.FeaturizeText("Features1", nameof(FilterStopWordsOutput.Tokens1)))
            .Append(context.Transforms.Text.FeaturizeText("Features2", nameof(FilterStopWordsOutput.Tokens2)))
            .Append(context.Transforms.Concatenate("Features", "Features1", "Features2"))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features")); // Algoritmo de clasificación

        // Entrenar el modelo
        var model = pipeline.Fit(trainData);

        // Ruta del archivo del modelo
        var modelPath = "qaModel.zip";

        // Guardar el modelo
        context.Model.Save(model, trainData.Schema, modelPath);
        Console.WriteLine("Modelo entrenado y guardado.");

        // Mostrar el tamaño del archivo del modelo
        var fileInfo = new FileInfo(modelPath);
        Console.WriteLine($"Tamaño del archivo del modelo: {fileInfo.Length / 1024.0} KB");

        // Mostrar predicciones con datos de entrenamiento
        ShowPredictions(context, model, trainingData);

        // Evaluar el modelo con datos de prueba
        EvaluateModelWithTestData(context, model);
    }

    static IEnumerable<T> LoadDataFromFolder<T>(string folderPath)
    {
        var files = Directory.GetFiles(folderPath, "*.json");

        var combinedData = new List<T>();
        foreach (var file in files)
        {
            var jsonData = File.ReadAllText(file);
            var data = JsonConvert.DeserializeObject<List<T>>(jsonData);
            combinedData.AddRange(data);
        }

        return combinedData;
    }

    static IEnumerable<T> LoadDataFromJson<T>(string filePath)
    {
        var jsonData = File.ReadAllText(filePath);
        return JsonConvert.DeserializeObject<List<T>>(jsonData);
    }

    static void ShowPredictions(MLContext context, ITransformer model, IEnumerable<QuestionPair> trainingData)
    {
        var predictor = context.Model.CreatePredictionEngine<QuestionPair, QuestionPrediction>(model);

        //foreach (var questionPair in trainingData)
        //{
        //    var prediction = predictor.Predict(questionPair);
        //    int matchingWordsCount = CountMatchingWords(questionPair.Question1, questionPair.Question2);
        //    // Ajustar el puntaje de la predicción basado en el número de palabras coincidentes
        //    prediction.PredictedLabel += matchingWordsCount * 0.1f;

        //    Console.WriteLine($"Question1: {questionPair.Question1}");
        //    Console.WriteLine($"Question2: {questionPair.Question2}");
        //    Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
        //    Console.WriteLine();
        //}
    }

    static void EvaluateModelWithTestData(MLContext context, ITransformer model)
    {
        // Cargar los datos de prueba desde JSON
        var testData = LoadDataFromJson<QuestionPair>("testData.json");

        // Convertir las listas a IDataView
        var testDataView = context.Data.LoadFromEnumerable(testData);

        // Crear el predictor
        var predictor = context.Model.CreatePredictionEngine<QuestionPair, QuestionPrediction>(model);

        // Mostrar predicciones
        foreach (var questionPair in testData)
        {
            var prediction = predictor.Predict(questionPair);
            int matchingWordsCount = CountMatchingWords(questionPair.Question1, questionPair.Question2);
            // Ajustar el puntaje de la predicción basado en el número de palabras coincidentes
            prediction.PredictedLabel += matchingWordsCount * 0.1f;

            Console.WriteLine($"Question1: {questionPair.Question1}");
            Console.WriteLine($"Question2: {questionPair.Question2}");
            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
            Console.WriteLine();
        }
    }

    static string[] FilterStopWords(string[] tokens)
    {
        var stopwords = new HashSet<string>(new[]
   {
        "el", "la", "los", "las", "de", "en", "y", "a", "que", "es", "con", "por", "como", "para", "un", "una", "al", "se",
        "él", "ella", "ellos", "ellas", "del", "á", "é", "í", "ó", "ú", "áéíóú", "cómo", "cuándo", "dónde", "qué", "qué"
        // Añade más palabras comunes acentuadas si es necesario
    });

        return tokens.Where(token => !stopwords.Contains(token.ToLower())).ToArray();
    }

    static int CountMatchingWords(string text1, string text2)
    {
        var tokens1 = FilterStopWords(Tokenize(text1));
        var tokens2 = FilterStopWords(Tokenize(text2));

        return tokens1.Intersect(tokens2).Count();
    }

    static string[] Tokenize(string text)
    {
        // Tokenizar el texto en palabras
        return text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
    }
}

// Clases de datos
public class QuestionPair
{
    public string Question1 { get; set; }
    public string Question2 { get; set; }
    public bool Label { get; set; } // Añadido para la clasificación
}

public class QuestionPrediction
{
    [ColumnName("Score")]
    public float PredictedLabel { get; set; }
}

// Clases para la eliminación de stopwords y ponderación
public class FilterStopWordsInput
{
    public string[] Tokens1 { get; set; }
    public string[] Tokens2 { get; set; }
}

public class FilterStopWordsOutput
{
    public string[] Tokens1 { get; set; }
    public string[] Tokens2 { get; set; }
}

public static class TokenWeights
{
    private static readonly Dictionary<string, float> Weights = new Dictionary<string, float>
    {
        { "sujeto", 2.0f },
        { "verbo", 5.5f },
        { "nombre", 1.2f },
        // Añadir más ponderaciones si es necesario
    };

    public static float GetWeight(string token)
    {
        return Weights.TryGetValue(token.ToLower(), out var weight) ? weight : 1.0f;
    }
}
