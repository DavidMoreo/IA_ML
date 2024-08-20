using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

public class TextCompletionInput
{
    public string InputText { get; set; }
    public string LabelText { get; set; }
}

public class TextCompletionOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedText { get; set; }
}

public class TextCompletionData
{
    public string InputText { get; set; }
    public string LabelText { get; set; }
}

public class TextCompletionModel
{
    // Datos predefinidos para completar frases
    private static readonly List<TextCompletionData> TrainingData = new List<TextCompletionData>
{
    new TextCompletionData { InputText = "El clima está", LabelText = "muy agradable hoy." },
    new TextCompletionData { InputText = "Me gusta mucho", LabelText = "la programación en C#." },
    new TextCompletionData { InputText = "El café es una bebida", LabelText = "popular en todo el mundo." },
    new TextCompletionData { InputText = "La inteligencia artificial", LabelText = "está transformando muchas industrias." },

    new TextCompletionData { InputText = "La casa tiene un jardín", LabelText = "hermoso y bien cuidado." },
    new TextCompletionData { InputText = "El software de edición de fotos", LabelText = "permite ajustar la exposición y el contraste." },
    new TextCompletionData { InputText = "Los libros de ciencia ficción", LabelText = "pueden ofrecer una visión fascinante del futuro." },
    new TextCompletionData { InputText = "La música clásica", LabelText = "es conocida por su complejidad y belleza." },

    new TextCompletionData { InputText = "La comida italiana", LabelText = "es famosa por sus pastas y pizzas." },
    new TextCompletionData { InputText = "El deporte más popular en el mundo", LabelText = "es el fútbol." },
    new TextCompletionData { InputText = "El entrenamiento en la mañana", LabelText = "puede aumentar tu energía durante el día." },
    new TextCompletionData { InputText = "Los parques naturales", LabelText = "ofrecen un refugio para la vida silvestre y son ideales para el senderismo." },

    new TextCompletionData { InputText = "Las redes sociales", LabelText = "han cambiado la forma en que nos comunicamos." },
    new TextCompletionData { InputText = "Un buen libro", LabelText = "puede ofrecer una escapada de la realidad." },
    new TextCompletionData { InputText = "La tecnología está avanzando rápidamente", LabelText = "y está cambiando todos los aspectos de nuestras vidas." },
    new TextCompletionData { InputText = "El aprendizaje automático", LabelText = "es un campo de la inteligencia artificial que enseña a las máquinas a aprender de los datos." },

    new TextCompletionData { InputText = "El cine de animación", LabelText = "a menudo utiliza técnicas avanzadas de gráficos por computadora." },
    new TextCompletionData { InputText = "Las vacaciones en la playa", LabelText = "son ideales para relajarse y disfrutar del sol." },
    new TextCompletionData { InputText = "El diseño gráfico", LabelText = "es fundamental para crear una comunicación visual efectiva." },
    new TextCompletionData { InputText = "La lectura es", LabelText = "una excelente forma de adquirir nuevos conocimientos y entretenerse." },

    new TextCompletionData { InputText = "Los animales en peligro de extinción", LabelText = "requieren esfuerzos especiales para su conservación." },
    new TextCompletionData { InputText = "Los videojuegos", LabelText = "ofrecen una forma interactiva de entretenimiento y desafío." },
    new TextCompletionData { InputText = "El arte moderno", LabelText = "puede desafiar las percepciones tradicionales del arte." },
    new TextCompletionData { InputText = "La cocina molecular", LabelText = "es una técnica culinaria que utiliza principios científicos para crear platos innovadores." },

    new TextCompletionData { InputText = "La meditación diaria", LabelText = "puede ayudar a reducir el estrés y mejorar el bienestar general." },
    new TextCompletionData { InputText = "El reciclaje es", LabelText = "esencial para la protección del medio ambiente." },
    new TextCompletionData { InputText = "Las aplicaciones móviles", LabelText = "han revolucionado la forma en que interactuamos con la tecnología." },
    new TextCompletionData { InputText = "El diseño web", LabelText = "es crucial para crear experiencias de usuario efectivas en internet." }
};


    // Método para entrenar el modelo
    public static void TrainModel()
    {
        var mlContext = new MLContext();

        // Cargar los datos de entrenamiento
        var data = mlContext.Data.LoadFromEnumerable(TrainingData);

        // Preprocesamiento de datos (Convertir texto en números)
        var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextCompletionData.InputText))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelText", nameof(TextCompletionData.LabelText)));

        // Algoritmo de clasificación multiclase
        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
            labelColumnName: "LabelText", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Entrenamiento del modelo
        var trainingPipeline = dataPipeline.Append(trainer);
        var model = trainingPipeline.Fit(data);

        // Guardar el modelo entrenado
        mlContext.Model.Save(model, data.Schema, "textCompletionModel.zip");
    }

    // Método para predecir el texto completado basado en la entrada
    public static string PredictCompletion(string inputText)
    {
        var mlContext = new MLContext();

        // Cargar el modelo entrenado
        var model = mlContext.Model.Load("textCompletionModel.zip", out var schema);

        // Crear un predictor
        var predictor = mlContext.Model.CreatePredictionEngine<TextCompletionInput, TextCompletionOutput>(model);

        // Predecir el texto completado
        var input = new TextCompletionInput { InputText = inputText };
        var result = predictor.Predict(input);

        return result.PredictedText;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo (esto lo haces una vez)
        TextCompletionModel.TrainModel();

        // Bucle infinito para permitir la generación de texto continuo
        while (true)
        {
            Console.WriteLine("Ingrese un texto para completar (o 'salir' para terminar):");
            var inputText = Console.ReadLine();

            if (string.Equals(inputText, "salir", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            // Usar el modelo para hacer una predicción
            string completion = TextCompletionModel.PredictCompletion(inputText);

            string list = inputText;

            foreach (var item in completion.Split(" "))
            {

                if (!inputText.Split(" ").Contains(item.ToLower()))
                {
                    list += " ";
                    list += item;
                }
            }

            Console.WriteLine($"Texto completado: {inputText} '{list}'");
        }
    }
}
