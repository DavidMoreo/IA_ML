////using System;
////using System.Collections.Generic;
////using System.Linq;
////using Microsoft.ML;
////using Microsoft.ML.Data;
////using Microsoft.ML.Transforms.Text;

////// Define las clases para los datos
////public class QAData
////{
////    public string Question { get; set; }
////    public string Answer { get; set; }
////}

////public class QuestionInput
////{
////    public string Question { get; set; }
////}

////public class QuestionPrediction
////{
////    [ColumnName("Features")]
////    public float[] Features { get; set; }
////}

////class Program
////{
////    static void Main(string[] args)
////    {
////        // Crear el contexto ML.NET
////        var context = new MLContext();

////        // Datos de ejemplo
////        var qaPairs = new[]
////        {
////            new QAData { Question = "¿Cuál es la capital de Francia?", Answer = "París" },
////                        new QAData { Question = "¿Viajaste en navidad la capital de Francia?", Answer = "París" },
////            new QAData { Question = "¿Cómo se llama el presidente de Estados Unidos?", Answer = "Joe Biden" },
////            new QAData { Question = "¿tiene memoria ram en pc?", Answer = "50 gb de ram" },
////            new QAData { Question = "¿tiene memoria rom en pc?", Answer = "50 gb de rom" },
////            new QAData { Question = "¿El producto tiene garantia?", Answer = "si tiene garantia pero solo de fabrica" },
////            new QAData { Question = "¿Cómo se llama el perro del presidente de Estados Unidos?", Answer = "perro Joe Biden" },
////            new QAData { Question = "¿Cuál es la moneda de Japón?", Answer = "Yen" },
////            new QAData { Question = "¿Quién es el autor de 'Cien años de soledad'?", Answer = "Gabriel García Márquez" },
////            new QAData { Question = "¿Cuál es la capital de Alemania?", Answer = "Berlín" },
////            new QAData { Question = "¿Cómo se llama el presidente de Rusia?", Answer = "Vladimir Putin" }
////        };

////        // Cargar los datos en un IDataView
////        var data = context.Data.LoadFromEnumerable(qaPairs);

////        // Configurar el pipeline de procesamiento
////        var pipeline = context.Transforms.Text.FeaturizeText("Features", "Question");

////        // Entrenar el modelo
////        var model = pipeline.Fit(data);

////        // Ingresar la pregunta del usuario
////        Console.Write("Por favor, ingrese su pregunta: ");
////        var customerQuestion = Console.ReadLine();

////        // Preparar los datos para la predicción
////        var input = new QuestionInput { Question = customerQuestion };
////        var inputData = context.Data.LoadFromEnumerable(new[] { input });

////        // Transformar la pregunta del usuario usando el modelo entrenado
////        var transformedData = model.Transform(inputData);
////        var customerQuestionFeatures = context.Data.CreateEnumerable<QuestionPrediction>(transformedData, reuseRowObject: false).First().Features;

////        // Codificar las preguntas predefinidas
////        var questionData = context.Data.LoadFromEnumerable(qaPairs);
////        var transformedQuestionData = model.Transform(questionData);
////        var questionFeatures = context.Data.CreateEnumerable<QuestionPrediction>(transformedQuestionData, reuseRowObject: false).ToArray();

////        // Calcular la similitud entre la pregunta del cliente y las predefinidas
////        var similarities = questionFeatures
////            .Select(f => CosineSimilarity(customerQuestionFeatures, f.Features))
////            .ToArray();

////        // Seleccionar la pregunta más similar
////        var mostSimilarIdx = similarities
////            .Select((similarity, index) => new { similarity, index })
////            .OrderByDescending(x => x.similarity)
////            .First()
////            .index;

////        // Mostrar la respuesta correspondiente
////        var bestMatchAnswer = qaPairs[mostSimilarIdx].Answer;
////        Console.WriteLine($"Respuesta: {bestMatchAnswer}");
////    }

////    // Función para calcular la similitud del coseno entre dos vectores
////    static float CosineSimilarity(float[] vec1, float[] vec2)
////    {
////        var dotProduct = vec1.Zip(vec2, (v1, v2) => v1 * v2).Sum();
////        var magnitude1 = (float)Math.Sqrt(vec1.Sum(v => v * v));
////        var magnitude2 = (float)Math.Sqrt(vec2.Sum(v => v * v));
////        return dotProduct / (magnitude1 * magnitude2);
////    }
////}


//using System;
//using System.Collections.Generic;
//using System.Linq;
//using Microsoft.ML;
//using Microsoft.ML.Data;
//using Microsoft.ML.Transforms.Text;

//// Define las clases para los datos
//public class QAData
//{
//    public string Question { get; set; }
//    public string Answer { get; set; }
//}

//public class QuestionInput
//{
//    public string Question { get; set; }
//}

//public class QuestionPrediction
//{
//    [ColumnName("Features")]
//    public float[] Features { get; set; }
//}

//class Program
//{
//    static void Main(string[] args)
//    {
//        var context = new MLContext();

//        // Datos de ejemplo
//        var qaPairs = new[]
//        {
//            new QAData { Question = "¿Cuál es la capital de Francia?", Answer = "París" },
//            new QAData { Question = "¿Cómo se llama el presidente de Estados Unidos?", Answer = "Joe Biden" },
//            new QAData { Question = "¿tiene memoria ram en pc?", Answer = "50 gb de ram" },
//            new QAData { Question = "¿tiene memoria rom en pc?", Answer = "50 gb de rom" },
//            new QAData { Question = "¿El producto tiene garantia?", Answer = "si tiene garantia pero solo de fabrica" },
//            new QAData { Question = "¿Cómo se llama el perro del presidente de Estados Unidos?", Answer = "perro Joe Biden" },
//            new QAData { Question = "¿Cuál es la moneda de Japón?", Answer = "Yen" },
//            new QAData { Question = "¿Quién es el autor de 'Cien años de soledad'?", Answer = "Gabriel García Márquez" },
//            new QAData { Question = "¿Cuál es la capital de Alemania?", Answer = "Berlín" },
//            new QAData { Question = "¿Cómo se llama el presidente de Rusia?", Answer = "Vladimir Putin" }
//        };

//        // Cargar los datos en un IDataView
//        var data = context.Data.LoadFromEnumerable(qaPairs);

//        // Configurar el pipeline de procesamiento
//        var pipeline = context.Transforms.Text.FeaturizeText("Features", "Question");

//        // Entrenar el modelo
//        var model = pipeline.Fit(data);

//        // Ingresar la pregunta del usuario
//        Console.Write("Por favor, ingrese su pregunta: ");
//        var customerQuestion = Console.ReadLine();

//        // Preparar los datos para la predicción
//        var input = new QuestionInput { Question = customerQuestion };
//        var inputData = context.Data.LoadFromEnumerable(new[] { input });

//        // Transformar la pregunta del usuario usando el modelo entrenado
//        var transformedData = model.Transform(inputData);
//        var customerQuestionFeatures = context.Data.CreateEnumerable<QuestionPrediction>(transformedData, reuseRowObject: false).First().Features;

//        // Codificar las preguntas predefinidas
//        var questionData = context.Data.LoadFromEnumerable(qaPairs);
//        var transformedQuestionData = model.Transform(questionData);
//        var questionFeatures = context.Data.CreateEnumerable<QuestionPrediction>(transformedQuestionData, reuseRowObject: false).ToArray();

//        // Calcular la similitud entre la pregunta del cliente y las predefinidas
//        var similarities = questionFeatures
//            .Select(f => CosineSimilarity(customerQuestionFeatures, f.Features))
//            .ToArray();

//        // Mostrar la similitud de cada pregunta predefinida con la pregunta del usuario
//        for (int i = 0; i < qaPairs.Length; i++)
//        {
//            Console.WriteLine($"Pregunta: {qaPairs[i].Question}");
//            Console.WriteLine($"Similitud: {similarities[i]}");
//            Console.WriteLine();
//        }

//        // Seleccionar la pregunta más similar
//        var mostSimilarIdx = similarities
//            .Select((similarity, index) => new { similarity, index })
//            .OrderByDescending(x => x.similarity)
//            .First()
//            .index;

//        // Mostrar la respuesta correspondiente
//        var bestMatchAnswer = qaPairs[mostSimilarIdx].Answer;
//        Console.WriteLine($"Respuesta: {bestMatchAnswer}");
//    }

//    // Función para calcular la similitud del coseno entre dos vectores
//    static float CosineSimilarity(float[] vec1, float[] vec2)
//    {
//        var dotProduct = vec1.Zip(vec2, (v1, v2) => v1 * v2).Sum();
//        var magnitude1 = (float)Math.Sqrt(vec1.Sum(v => v * v));
//        var magnitude2 = (float)Math.Sqrt(vec2.Sum(v => v * v));
//        return dotProduct / (magnitude1 * magnitude2);
//    }
//}
