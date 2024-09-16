using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

// Clase para representar los datos de entrenamiento
public class MessageData
{
    public string Text { get; set; }
    public string Intent { get; set; }  // Aquí almacenamos la intención
}

// Clase de entrada para ML.NET
public class MessageDataInput
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1)]
    public string Intent { get; set; }
}

// Clase de salida para ML.NET
public class MessageDataOutput
{
    [ColumnName("Score")]
    public float[] Score { get; set; }  // Cambiar a un array de floats para manejar el vector

    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }
}

public class IntentModel
{
    // Método para cargar datos desde archivos JSON en una carpeta
    public static List<MessageData> LoadDataFromFolder(string folderPath)
    {
        var trainingData = new List<MessageData>();

        foreach (var filePath in Directory.GetFiles(folderPath, "*.json"))
        {
            var jsonData = File.ReadAllText(filePath);
            var messageDataList = JsonConvert.DeserializeObject<List<MessageData>>(jsonData);

            // Filtrar las palabras vacías de los textos
            foreach (var message in messageDataList)
            {
                message.Text = string.Join(" ", FilterStopWords(message.Text.Split(' ')));
            }

            trainingData.AddRange(messageDataList);
        }

        return trainingData;
    }

    // Método para entrenar el modelo
    public static async void TrainModel(string folderPath)
    {
        // Cargar los datos
        //var trainingData = LoadDataFromFolder(folderPath);


        var trainingData = await GetHttp();

        // Crear el contexto de ML.NET
        var mlContext = new MLContext();

        // Convertir los datos en un IDataView
        var data = mlContext.Data.LoadFromEnumerable(trainingData);

        // Preprocesamiento de datos (Convertir texto en números)
        var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(MessageData.Text))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(MessageData.Intent)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        // Algoritmo de clasificación multicategoría
        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        // Entrenamiento del modelo
        var trainingPipeline = dataPipeline.Append(trainer);
        var model = trainingPipeline.Fit(data);

        // Guardar el modelo entrenado
        mlContext.Model.Save(model, data.Schema, "intentModel.zip");
    }

    public async static Task<List<MessageData>> GetHttp()
    {
        HttpClient client = new HttpClient();
        var list = new List<MessageData>();

        client.BaseAddress = new Uri("https://localhost:7049/");

        try
        {
         var http = await client.GetAsync("Chat/GetModelToTrainIntent");
            var respose = await http.Content.ReadAsStringAsync();
            list = JsonConvert.DeserializeObject<List<MessageData>>(respose);
            Console.WriteLine("Cantidad de registro :"+ list.Count);
        }
        catch (Exception ex)
        {

            throw;
        }
        return list;

    }


    // Método para predecir la intención basada en un mensaje
    public static string PredictIntent(string text)
    {
        var mlContext = new MLContext();

        // Filtrar palabras vacías del mensaje
        var filteredText = string.Join(" ", FilterStopWords(text.Split(' ')));

        // Cargar el modelo entrenado
        var model = mlContext.Model.Load("intentModel.zip", out var schema);

        // Crear un predictor
        var predictor = mlContext.Model.CreatePredictionEngine<MessageDataInput, MessageDataOutput>(model);

        // Predecir la intención
        var input = new MessageDataInput { Text = filteredText };
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

        return tokens.Where(token => !stopwords.Contains(token.ToLower())).ToArray();
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo (esto lo haces una vez)
        IntentModel.TrainModel("trainingData/");

        // Bucle infinito para permitir predicciones continuas
        while (true)
        {
            Console.WriteLine("Ingrese su mensaje (o 'salir' para terminar):");
            var text = Console.ReadLine();

            if (string.Equals(text, "salir", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            // Usar el modelo para predecir la intención del mensaje
            string intent = IntentModel.PredictIntent(text);

            Console.WriteLine($"Intención detectada: '{intent}'");
        }
    }
}
