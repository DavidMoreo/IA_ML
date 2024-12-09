using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

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
        public float Score { get; set; } // Puntaje de confianza
    }

    // Clase para datos desde la API
    public class QuestionPair
    {
        public string Question { get; set; }
        public string Answer { get; set; }
        public bool Label { get; set; }
    }

    // Ruta del modelo entrenado
    private static readonly string ModelPath = "../../../../MODEL/productDetectionModel.zip";

    // Lista de palabras comunes que se deben ignorar
    private static readonly HashSet<string> StopWords = new HashSet<string>
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "soy", "eres", "es",
        "somos", "son", "y", "o", "de", "a", "que", "en", "con", "por", "para", "mi", "tu",",", "."
    };

    // Obtener datos desde una API
    public async static Task<List<QuestionPair>> GetHttp()
    {
        HttpClient client = new HttpClient();
        var list = new List<QuestionPair>();
        client.BaseAddress = new Uri("https://compraenmiciudad.com/");

        try
        {
            var http = await client.GetAsync("Chat/GetTrainIaProductName");
            var response = await http.Content.ReadAsStringAsync();
            list = JsonConvert.DeserializeObject<List<QuestionPair>>(response);
            Console.WriteLine("Cantidad de registros obtenidos: " + list.Count);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error al obtener datos: {ex.Message}");
        }
        return list;
    }

    // Entrenar y guardar el modelo
    public static async Task TrainAndSaveModel()
    {
        var context = new MLContext();

        // Datos iniciales de entrenamiento
        var trainingData = new List<WordData>();
        

        Console.WriteLine("Obteniendo datos adicionales...");
        var list = await GetHttp();

        if (list != null && list.Any())
        {
            trainingData.AddRange(list.Select(l => new WordData
            {
                Word = l.Answer.ToLowerInvariant(),
                IsProduct = l.Label
            }));
        }

        Console.WriteLine("Datos cargados. Entrenando modelo...");

        // Convertir los datos a IDataView
        var data = context.Data.LoadFromEnumerable(trainingData);

        // Crear el pipeline de entrenamiento
        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(WordData.Word))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("IsProduct", "Features"));

        // Entrenar el modelo
        var model = pipeline.Fit(data);

        // Guardar el modelo
        context.Model.Save(model, data.Schema, ModelPath);
        Console.WriteLine("Modelo entrenado y guardado en: " + ModelPath);
    }

    // Predicción de productos
    public static (bool IsProduct, float Score) PredictProduct(string word)
    {
        var context = new MLContext();

        // Cargar el modelo entrenado
        var model = context.Model.Load(ModelPath, out _);

        // Crear el predictor
        var predictor = context.Model.CreatePredictionEngine<WordData, WordPrediction>(model);

        // Normalizar la palabra a minúsculas
        word = word.ToLower();

        // Realizar la predicción
        var prediction = predictor.Predict(new WordData { Word = word });

        return (prediction.IsProduct, prediction.Score); // Retorna la predicción y el puntaje
    }

    // Detectar productos en una oración
    public static void DetectProductsInSentence(string sentence)
    {
        var words = sentence.Split(' ');

        foreach (var word in words)
        {
            var normalizedWord = word.ToLower();
            if (StopWords.Contains(normalizedWord))
            {
               // Console.WriteLine($"Palabra: {normalizedWord} - Ignorada (común)");
                continue;
            }

            var (isProduct, score) = PredictProduct(normalizedWord);
           if(isProduct) Console.WriteLine($"Palabra: {normalizedWord} - - Confianza: {score:F4}");
        }
    }

    // Método principal
    public static async Task Main(string[] args)
    {
        Console.WriteLine("Entrenando modelo...");
        await TrainAndSaveModel();

        Console.WriteLine("Probando predicciones...");
        string descripcionProductos = "Los productos tecnológicos actuales destacan por su innovación y funcionalidad. Desde teléfonos inteligentes como el iPhone 15, Samsung Galaxy S24, y Google Pixel 8, con pantallas de alta resolución y cámaras avanzadas, hasta laptops ultradelgadas como la MacBook Air M2, Dell XPS 13, y Lenovo ThinkPad X1 Carbon, con baterías de larga duración. Además, las tablets como el iPad Pro y Samsung Galaxy Tab S9 ofrecen versatilidad y portabilidad para el trabajo y el entretenimiento. Cada detalle está diseñado para ofrecer una experiencia de usuario superior. " +
  "Estas características no solo facilitan la productividad, sino que también transforman la manera en que nos conectamos y disfrutamos del entretenimiento diario, ya sea viendo contenido en streaming en dispositivos como el Amazon Fire TV Stick o jugando con consolas como la PlayStation 5 o Xbox Series X. " +
  "Además, los nuevos dispositivos incorporan tecnologías de vanguardia, como inteligencia artificial y conectividad 5G, para optimizar su rendimiento. Por ejemplo, los asistentes de voz integrados en productos como el Amazon Echo o Google Nest Audio permiten realizar tareas de manera más eficiente, mientras que la compatibilidad con aplicaciones avanzadas amplía las posibilidades de uso. Estas características demuestran el compromiso de las marcas por satisfacer las necesidades cambiantes de los consumidores modernos.";
        DetectProductsInSentence(descripcionProductos.Replace("á","a").Replace("é", "e").Replace("í", "i").Replace("ó", "o").Replace("ú", "u"));
    }
}
