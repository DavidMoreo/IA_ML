using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

public class Product
{
    public string Name { get; set; }
}

public class ProductInput
{
    [LoadColumn(0)]
    public string Name { get; set; }
}

public class ProductPrediction
{
    [ColumnName("Score")]
    public float Score { get; set; }
}

public class ProductModel
{
    private static MLContext mlContext = new MLContext();
    private static ITransformer model;

    // Cargar datos desde archivos JSON
    public static List<Product> LoadDataFromFolder(string folderPath)
    {
        var productData = new List<Product>();

        foreach (var filePath in Directory.GetFiles(folderPath, "*.json"))
        {
            var jsonData = File.ReadAllText(filePath);
            var products = JsonConvert.DeserializeObject<List<Product>>(jsonData);
            productData.AddRange(products);
        }

        return productData;
    }

    // Entrenamiento del modelo
    public static void TrainModel(string folderPath)
    {
        var trainingData = LoadDataFromFolder(folderPath);

        // Convertir los datos en un IDataView
        var data = mlContext.Data.LoadFromEnumerable(trainingData);

        // Preprocesamiento de datos
        var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(Product.Name))
            .Append(mlContext.Transforms.Concatenate("Features", "Features"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Name"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("Name"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Features", maximumNumberOfIterations: 100);

        model = dataPipeline.Fit(data);
        mlContext.Model.Save(model, data.Schema, "productModel.zip");
    }

    // Predicción del producto más relevante
    public static string PredictMostRelevantProduct(string userInput)
    {
        var products = LoadDataFromFolder("trainingData/");
        var input = new ProductInput { Name = userInput };
        var predictor = mlContext.Model.Load("productModel.zip", out var schema);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductInput, ProductPrediction>(predictor);

        var mostRelevantProduct = products
            .Select(product => new
            {
                Product = product,
                Score = CalculateSimilarity(userInput, product.Name)
            })
            .OrderByDescending(x => x.Score)
            .FirstOrDefault();

        return mostRelevantProduct?.Product.Name;
    }

    // Método para calcular la similitud (puede ser ajustado)
    private static float CalculateSimilarity(string input, string productName)
    {
        // Simplificación: devuelve una puntuación basada en la coincidencia de palabras.
        var inputWords = input.Split(' ');
        var productWords = productName.Split(' ');
        var commonWordsCount = inputWords.Intersect(productWords).Count();
        return commonWordsCount / (float)inputWords.Length;
    }
}

class Program
{
    static void Main(string[] args)
    {
        var folderPath = "trainingData/";

        // Entrenar el modelo
        ProductModel.TrainModel(folderPath);

        // Bucle para permitir preguntas continuas
        while (true)
        {
            Console.WriteLine("Ingrese el nombre del producto (o 'salir' para terminar):");
            var name = Console.ReadLine();

            if (string.Equals(name, "salir", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            var mostRelevantProduct = ProductModel.PredictMostRelevantProduct(name);
            Console.WriteLine($"Producto más relevante:");
            Console.WriteLine($"Nombre: {mostRelevantProduct}");
        }
    }
}
