using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

class Program
{
    public class SentenceData
    {
        public string Sentence { get; set; }
        public string Category { get; set; }  // La categoría a predecir
    }

    public class SentencePrediction
    {
        public string PredictedCategory { get; set; }
    }

    static void Main()
    {
        var context = new MLContext();

        // Lista de frases con categorías definidas
        var sentences = new List<SentenceData>
        {
            new SentenceData { Sentence = "Quiero comprar un teléfono móvil", Category = "Celular" },
            new SentenceData { Sentence = "Estoy buscando un celular", Category = "Celular" },
            new SentenceData { Sentence = "¿Tienen algún teléfono inteligente?", Category = "Celular" },
            new SentenceData { Sentence = "¿Dónde puedo encontrar un smartphone?", Category = "Celular" },
            new SentenceData { Sentence = "Necesito un televisor de 55 pulgadas", Category = "Televisor" },
            new SentenceData { Sentence = "Busco un televisor 4K", Category = "Televisor" },
            new SentenceData { Sentence = "Busco un tv 4K", Category = "Televisor" },
        };

        // Crear el conjunto de datos de entrenamiento
        var data = context.Data.LoadFromEnumerable(sentences);

        // Convertir las frases en vectores numéricos usando TextFeaturizingEstimator
        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentenceData.Sentence))
            .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(SentenceData.Category)))
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedCategory", "PredictedLabel"));

        // Entrenar el modelo
        var model = pipeline.Fit(data);

        // Realizar las predicciones
        var predictions = model.Transform(data);

        // Mostrar los resultados
        var result = context.Data.CreateEnumerable<SentencePrediction>(predictions, reuseRowObject: false).ToList();

        for (int i = 0; i < result.Count; i++)
        {
            Console.WriteLine($"Frase: {sentences[i].Sentence} -> Categoría Predicha: {result[i].PredictedCategory}");
        }

        // Agregar nuevas frases indefinidamente por consola
        while (true)
        {
            Console.WriteLine("Ingresa una frase (o 'salir' para terminar): ");
            var userInput = Console.ReadLine();
            if (userInput.ToLower() == "salir") break;

            var newSentence = new List<SentenceData> { new SentenceData { Sentence = userInput } };
            var newData = context.Data.LoadFromEnumerable(newSentence);
            var newPredictions = model.Transform(newData);
            var newResult = context.Data.CreateEnumerable<SentencePrediction>(newPredictions, reuseRowObject: false).ToList();

            foreach (var pred in newResult)
            {
                Console.WriteLine($"Frase: {userInput} -> Categoría Predicha: {pred.PredictedCategory}");
            }
        }
    }
}
