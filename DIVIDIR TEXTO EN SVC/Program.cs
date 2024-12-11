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
                Word = l.Answer.Replace("á", "a").Replace("é", "e").Replace("í", "i").Replace("ó", "o").Replace("ú", "u").ToLower(),
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
        string descripcionProductos =
"- Kits básicos de aprendizaje de Arduino\n" +
"- Robots móviles para proyectos educativos\n" +
"- Drones controlados por Arduino\n" +
"- Brazos robóticos programables\n" +
"- Sensores de proximidad para robots\n" +
"- Módulos de control remoto por Bluetooth\n" +
"- Carros seguidores de línea\n" +
"- Proyectos de domótica con Arduino\n" +
"- Impresoras 3D controladas por microcontroladores\n" +
"- Robots bípedos educativos\n" +
"- Módulos de comunicación Wi-Fi para IoT\n" +
"- Sistemas de riego automatizado con Arduino\n" +
"- Vehículos autónomos para investigación\n" +
"- Placas de expansión para Arduino\n" +
"- Proyectos de reconocimiento de voz con Arduino\n" +
"- Sensores ultrasónicos para evitar obstáculos\n" +
"- Kits de drones autónomos\n" +
"- Cámaras para visión robótica\n" +
"- Robots recolectores de basura\n" +
"- Proyectos de iluminación inteligente\n" +
"- Relojes inteligentes basados en microcontroladores\n" +
"- Estaciones meteorológicas controladas por Arduino\n" +
"- Sistemas de monitoreo de calidad del aire\n" +
"- Interfaces de control por gestos\n" +
"- Robots araña programables\n" +
"- Sensores de temperatura y humedad para proyectos\n" +
"- Módulos GPS para rastreo\n" +
"- Robots limpiadores de paneles solares\n" +
"- Proyectos de brazo robótico para ensamblaje\n" +
"- Kits de robots de competencia\n" +
"- Robots submarinos de exploración\n" +
"- Sensores de pulso y oxígeno para aplicaciones médicas\n" +
"- Proyectos de control de motores con Arduino\n" +
"- Controladores de servomotores para robótica\n" +
"- Plataformas de vehículos controladas por aplicaciones móviles\n" +
"- Módulos RFID para identificación automática.";



        DetectProductsInSentence(descripcionProductos.Replace("á","a").Replace("é", "e").Replace("í", "i").Replace("ó", "o").Replace("ú", "u"));
    }
}
