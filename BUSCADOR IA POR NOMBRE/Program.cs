using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
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
    [ColumnName("PredictedLabel")]
    public string Answer { get; set; }

    public float[] Score { get; set; }  // Confidence scores for each class
}

public class QuestionAnswerModel
{
    private static readonly string ModelPath = "../../../../MODEL/productDetectionModel.zip";
    private static readonly HttpClient client = new HttpClient
    {
        BaseAddress = new Uri("https://localhost:7049/") // Change the URL as needed
    };

    // Method to train the model
    public static async Task TrainModelAsync(string folderPath)
    {
        try
        {
            // Load training data
            var trainingData = await GetTrainingDataAsync();

            // Create ML.NET context
            var mlContext = new MLContext();

            // Convert data to IDataView
            var data = mlContext.Data.LoadFromEnumerable(trainingData);

            // Data preprocessing (Convert text to numeric features)
            var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(QuestionPair.Question))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(QuestionPair.Answer)))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Multiclass classification algorithm
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Model training
            Console.WriteLine("Training model...");
            var trainingPipeline = dataPipeline.Append(trainer);
            var model = trainingPipeline.Fit(data);

            // Save the trained model
            Console.WriteLine("Saving model...");
            mlContext.Model.Save(model, data.Schema, ModelPath);
            Console.WriteLine("Model saved.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred during model training: {ex.Message}");
            // Log the exception if necessary
        }
    }

    // Method to fetch training data from an API
    private static async Task<List<QuestionPair>> GetTrainingDataAsync()
    {
        try
        {
            var response = await client.GetAsync("Chat/GetTrainIaProductName");
            response.EnsureSuccessStatusCode();  // Ensures that a successful HTTP status code is received
            var jsonResponse = await response.Content.ReadAsStringAsync();
            var list = JsonConvert.DeserializeObject<List<QuestionPair>>(jsonResponse);
            Console.WriteLine($"Number of records fetched: {list.Count}");
            return list;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred while fetching training data: {ex.Message}");
            // Optionally, rethrow or handle the exception as appropriate
            return new List<QuestionPair>();
        }
    }

    // Method to predict the answer based on a question
    public static (string Answer, float Score) PredictAnswer(string question)
    {
        try
        {
            var mlContext = new MLContext();

            // Preprocess the question (remove stop words)
            var filteredQuestion = string.Join(" ", FilterStopWords(question.Split(' ')));

            // Load the trained model
            var model = mlContext.Model.Load(ModelPath, out var schema);

            // Create a prediction engine
            var predictor = mlContext.Model.CreatePredictionEngine<QuestionPairInput, QuestionPairOutput>(model);

            // Predict the answer
            var input = new QuestionPairInput { Question = filteredQuestion };
            var result = predictor.Predict(input);

            // Get the highest score and its corresponding predicted answer
            var maxScore = result.Score.Max();
            var predictedAnswer = result.Answer;

            return (predictedAnswer, maxScore);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred during prediction: {ex.Message}");
            return (string.Empty, 0f); // Return default values in case of an error
        }
    }

    // Method to filter out stop words from a set of tokens
    private static string[] FilterStopWords(string[] tokens)
    {
        var stopwords = new HashSet<string>(new[]
        {
            "el", "la", "los", "las", "de", "en", "y", "a", "que", "es", "con", "por", "como", "para", "un", "una", "al", "se",
            "él", "ella", "ellos", "ellas", "del", "á", "é", "í", "ó", "ú" // Add more stop words as needed
        });

        return tokens.Where(token => !stopwords.Contains(token.ToLower())).ToArray();
    }
}


class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo (esto lo haces una vez)
        QuestionAnswerModel.TrainModelAsync("trainingData/");

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
            var (answer, score) = QuestionAnswerModel.PredictAnswer(question);

            Console.WriteLine($"Respuesta: '{answer}' ");
            Console.WriteLine($" : {score}");
        }
    }
}
