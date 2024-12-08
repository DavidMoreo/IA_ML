using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

public class ProductDetector
{
    // Clase para los datos de entrada
    public class WordData
    {
        public string Word { get; set; }
        public bool IsProduct { get; set; }
    }

    // Clase para las predicciones
    public class WordPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsProduct { get; set; }
    }

    public class QuestionPair
    {
        public string Question { get; set; }
        public string Answer { get; set; }

        public bool Label { get; set; }
    }

    // Ruta del modelo entrenado
    private static readonly string ModelPath = "../../../../MODEL/productDetectionModel.zip";


    public async static Task<List<QuestionPair>> GetHttp()
    {
        HttpClient client = new HttpClient();
        var list = new List<QuestionPair>();

        client.BaseAddress = new Uri("https://compraenmiciudad.com/");

        try
        {
            var http = await client.GetAsync("Chat/GetTrainIaProductName");
            var respose = await http.Content.ReadAsStringAsync();
            list = JsonConvert.DeserializeObject<List<QuestionPair>>(respose);
            Console.WriteLine("Cantidad de registro :" + list.Count);
        }
        catch (Exception ex)
        {

            throw;
        }
        return list;

    }


    public static async void TrainAndSaveModel()
    {
        var context = new MLContext();

        // Datos de entrenamiento: palabras etiquetadas como productos o no productos
        var trainingData = new List<WordData>
        {
            new WordData { Word = "yo", IsProduct = false },
            new WordData { Word = "quiero", IsProduct = false },
            new WordData { Word = "celular", IsProduct = true },
            new WordData { Word = "laptop", IsProduct = true },
            new WordData { Word = "cargador", IsProduct = true },
            new WordData { Word = "bicicleta", IsProduct = true },
            new WordData { Word = "silla", IsProduct = true },
            new WordData { Word = "televisor", IsProduct = true },
            new WordData { Word = "gato", IsProduct = false },
            new WordData { Word = "caminar", IsProduct = false },
            new WordData { Word = "parque", IsProduct = false }
        };

        Console.WriteLine("Buscando data.");
        var list = await GetHttp();

        if (list!=null)
        {
            trainingData =(from l in list
             select
             new WordData
             {
                 Word = l.Answer,
                 IsProduct = l.Label
             }).ToList();

           
        }
        Console.WriteLine("data encontrada.");
        // Convertir los datos a IDataView
        var data = context.Data.LoadFromEnumerable(trainingData);

        // Crear el pipeline de entrenamiento
        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(WordData.Word))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("IsProduct", "Features"));

        Console.WriteLine("Entrenan do.");
        // Entrenar el modelo
        var model = pipeline.Fit(data);
        Console.WriteLine("Fin entrenbamiento  do.");
        // Guardar el modelo
        Console.WriteLine("Guardando.");
        context.Model.Save(model, data.Schema, ModelPath);
        Console.WriteLine("Guardado.");


        Console.WriteLine("Modelo entrenado y guardado.");

      
    }

    public static bool PredictProduct(string word)
    {
        var context = new MLContext();

        // Cargar el modelo entrenado
        var model = context.Model.Load(ModelPath, out _);

        // Crear el predictor
        var predictor = context.Model.CreatePredictionEngine<WordData, WordPrediction>(model);

        // Realizar la predicción
        var prediction = predictor.Predict(new WordData { Word = word });

        return prediction.IsProduct; // Retorna si la palabra es un producto
    }

    public static void DetectProductsInSentence(string sentence)
    {
        var words = sentence.Split(' ');

        foreach (var word in words)
        {
            bool isProduct = PredictProduct(word);
            Console.WriteLine($"Palabra: {word} - ¿Es producto?: {isProduct}");
        }
    }



    public static async void  LoadData()
    {
        TrainAndSaveModel();
    }

    public static  void Main(string[] args)
    {
        // Entrenar y guardar el modelo
        //LoadData();

        //while (true)
        //{ }

        // Pruebas de detección de productos
        Console.WriteLine("\nDetección de productos en las frases:");
        DetectProductsInSentence("Yo quiero un celular");
        DetectProductsInSentence("Ellos Quieren laptops");
        DetectProductsInSentence("Necesito una silla");
        DetectProductsInSentence("Voy al parque a caminar");
    }
}
