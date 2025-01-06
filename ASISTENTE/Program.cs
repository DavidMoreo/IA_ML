using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json;
using static Program;

class Program
{

    static async Task Main()
    {
        var data = new LoadData();
        data.LoadAsync();
        while (true) { }
    }


}


class LoadData
{
    public class ModelToTrainAssistan
    {
        public string Question { get; set; }
        public string Answer { get; set; }
        public bool Label { get; set; }
    }

    public class SentenceData
    {
        public string Question { get; set; }
        public string Category { get; set; }  // La categoría a predecir
    }

    public class SentencePrediction
    {
        [ColumnName("PredictedCategory")]
        public string PredictedCategory { get; set; }
        public float[] Score { get; set; }
    }

    public class NameConcept
    {
        public string? Name { get; set; }
        public string? Value { get; set; }
        public Guid? Id { get; set; }
    }

    public string Route { get { return "../../../../MODEL/ModelIAAssistant.zip"; } }
    public async Task LoadAsync()
    {
        var context = new MLContext();

        // Obtener datos desde la API usando el método GetHttp
        var nameConcepts = await GetHttp();

        // Convertir los datos de NameConcept a SentenceData
        var sentences = MapToSentenceData(nameConcepts);

        if (sentences == null || sentences.Count == 0)
        {
            Console.WriteLine("No se pudieron obtener datos de entrenamiento.");
            return;
        }

        // Crear el conjunto de datos de entrenamiento
        var data = context.Data.LoadFromEnumerable(sentences);

        //// Construir la pipeline
        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentenceData.Question))
            .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(SentenceData.Category)))
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features", null, null, 10))
            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedCategory", "PredictedLabel"));

        //    var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentenceData.Question))
        //.Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(SentenceData.Category)))
        //.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        //.Append(context.Transforms.Conversion.MapKeyToValue("PredictedCategory", "PredictedLabel"));


        //// Construir el pipeline
        //var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentenceData.Question))
        //    .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(SentenceData.Category)))
        //    .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features", null, null, 10))
        //    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedCategory", "PredictedLabel"))
        //    .Append(context.Clustering.Trainers.KMeans(
        //        featureColumnName: "Features")); // Número de clústeres deseado



        // Entrenar el modelo
        Console.WriteLine("Entrenando modelo. " + sentences.Count());
        var model = pipeline.Fit(data);

        Console.WriteLine("Guardando el modelo entrenado.");
        context.Model.Save(model, data.Schema, Route);
        Console.WriteLine("Modelo guardado como 'modelo_entrenado.zip'.");

        // Realizar predicciones en tiempo real


        Console.WriteLine("Modelo entrenado. Ahora puedes hacer predicciones.");


        while (true)
        {
            Console.Write("> ");
            Console.WriteLine("Escribe una pregunta o 'salir' para terminar:");
            var input = Console.ReadLine();
            if (string.IsNullOrEmpty(input) || input.ToLower() == "salir")
            {
                break;
            }
            var predictionEngine = context.Model.CreatePredictionEngine<SentenceData, SentencePrediction>(model);
            var prediction = predictionEngine.Predict(new SentenceData { Question = input });
            var max = prediction.Score.Max();
            Console.WriteLine($"Predicción: {prediction.PredictedCategory}  Score : {max}");
        }
    }
    public async static Task<List<ModelToTrainAssistan>> GetHttp()
    {
        HttpClient client = new HttpClient();
        var list = new List<ModelToTrainAssistan>();

        client.BaseAddress = new Uri("https://localhost:7049/");
        try
        {
            var http = await client.GetAsync("Chat/GetModelToTrainAssistan");
            var response = await http.Content.ReadAsStringAsync();
            list = JsonConvert.DeserializeObject<List<ModelToTrainAssistan>>(response);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error al obtener datos: {ex.Message}");
        }

        return list;
    }
    //  var responseChat = await _Chat.GetAllChatIaConfigByType(config, tblChatTypeModelIa.SearchProductsByName);

    public static List<SentenceData> MapToSentenceData(List<ModelToTrainAssistan> nameConcepts)
    {
        var sentenceData = new List<SentenceData>();

        foreach (var concept in nameConcepts)
        {
            if (!string.IsNullOrEmpty(concept.Question) && !string.IsNullOrEmpty(concept.Question))
            {
                sentenceData.Add(new SentenceData
                {
                    Question = concept.Question,
                    Category = concept.Answer

                });
            }
        }

        return sentenceData;
    }
}