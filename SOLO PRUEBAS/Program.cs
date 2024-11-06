using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

public class WordPositionInput
{
    public string InputText { get; set; }
    public string Word { get; set; }
    public string Position { get; set; } // "antes" o "después"

}

public class WordPositionOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedPosition { get; set; } // "antes" o "después"
}

public class WordPositionData
{
    public string InputText { get; set; }
    public string Word { get; set; }
    public string Position { get; set; } // "antes" o "después"
}

public class WordPositionModel
{
    // Datos de entrenamiento para determinar la posición de la palabra
    private static readonly List<WordPositionData> TrainingData = new List<WordPositionData>
    {
        new WordPositionData { InputText = "que clima está", Word = "agradable", Position = "después" },
        new WordPositionData { InputText = "yo", Word = "pienzo", Position = "después" },
        new WordPositionData { InputText = "el", Word = "clima", Position = "después" },
        new WordPositionData { InputText = "clima esta", Word = "raro", Position = "después" },
        // Agrega más ejemplos para mejorar el aprendizaje
    };

    // Método para entrenar el modelo
    public static void TrainModel()
    {
        var mlContext = new MLContext();

        // Cargar los datos de entrenamiento
        var data = mlContext.Data.LoadFromEnumerable(TrainingData);

        // Preprocesamiento de los datos (Convertir texto en números)
        var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(WordPositionData.InputText))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Position", nameof(WordPositionData.Position)));

        // Algoritmo de clasificación multiclase para la predicción de la posición
        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
            labelColumnName: "Position", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Entrenamiento del modelo
        var trainingPipeline = dataPipeline.Append(trainer);
        var model = trainingPipeline.Fit(data);

        // Guardar el modelo entrenado
        mlContext.Model.Save(model, data.Schema, "wordPositionModel.zip");
    }

    // Método para predecir si la palabra debe ir antes o después
    public static string PredictWordPosition(string inputText, string word)
    {
        var mlContext = new MLContext();

        // Cargar el modelo entrenado
        var model = mlContext.Model.Load("wordPositionModel.zip", out var schema);

        // Crear un predictor
        var predictor = mlContext.Model.CreatePredictionEngine<WordPositionInput, WordPositionOutput>(model);

        // Predecir la posición de la palabra
        var input = new WordPositionInput { InputText = inputText, Word = word };
        var result = predictor.Predict(input);

        return result.PredictedPosition;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo (esto se hace una vez)
        WordPositionModel.TrainModel();

        // Bucle para permitir al usuario probar la predicción
        while (true)
        {
            Console.WriteLine("Ingrese un texto para completar (o 'salir' para terminar):");
            var inputText = Console.ReadLine();

            if (string.Equals(inputText, "salir", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            Console.WriteLine("Ingrese una palabra que desea agregar al texto:");
            var word = Console.ReadLine();

            // Usar el modelo para predecir si la palabra va antes o después
            string position = WordPositionModel.PredictWordPosition(inputText, word);

            // Insertar la palabra en la posición correcta
            string completedText = position == "antes"
                ? word + " " + inputText
                : inputText + " " + word;

            Console.WriteLine($"Texto completado: '{completedText}'");
        }
    }
}
