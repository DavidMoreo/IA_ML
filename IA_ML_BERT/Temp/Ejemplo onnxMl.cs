//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using Microsoft.ML;
//using Microsoft.ML.Data;
//using Microsoft.ML.Transforms.Onnx;

//class Program
//{
//    static void Main(string[] args)
//    {
//        // Ruta al modelo ONNX y al diccionario de tokens
//        string modelPath = "mlp_model.onnx";
//        string tokenizerPath = "tokenizer_vocab.csv";

//        // Crear el contexto ML
//        var context = new MLContext();

//        // Cargar el modelo ONNX
//        var pipeline = context.Transforms.ApplyOnnxModel(
//            modelFile: modelPath,
//            outputColumnNames: new[] { "output" },
//            inputColumnNames: new[] { "input" }
//        );

//        // Crear un conjunto de datos vacío
//        var emptyData = context.Data.LoadFromEnumerable(new List<InputData>());

//        // Ajustar el modelo ONNX a los datos vacíos
//        var model = pipeline.Fit(emptyData);

//        // Crear el motor de predicción
//        var predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(model);

//        // Definir el contexto y la pregunta
//        string contextText = "El producto es un smartphone con una pantalla de 6.5 pulgadas...";
//        string questionText = "¿Cuánta memoria soporta?";

//        // Preparar los datos de entrada
//        var inputData = PrepareInputData(contextText, questionText, tokenizerPath);

//        // Realizar la predicción
//        var output = predictionEngine.Predict(inputData);

//        // Decodificar la salida a texto
//        string decodedText = DecodeTokens(output.Output, tokenizerPath);
//        Console.WriteLine($"Respuesta generada: {decodedText}");
//    }

//    static InputData PrepareInputData(string context, string question, string tokenizerPath)
//    {
//        var tokenizer = LoadVocabulary(tokenizerPath);
//        var tokens = TokenizeText($"{context} {question}", tokenizer);

//        // Ajustar tamaño del vector según el modelo
//        int vectorSize = 8; // Mantén esto como 512
//        var paddedTokens = new float[vectorSize];
//        for (int i = 0; i < Math.Min(tokens.Count, vectorSize); i++)
//        {
//            paddedTokens[i] = tokens[i];
//        }

//        return new InputData
//        {
//            Input = paddedTokens
//        };
//    }

//    static List<float> TokenizeText(string text, Dictionary<string, int> tokenizer)
//    {
//        var tokens = text.Split(' ');
//        var tokenIds = new List<float>();

//        foreach (var token in tokens)
//        {
//            if (tokenizer.TryGetValue(token, out int tokenId))
//            {
//                tokenIds.Add(tokenId);
//            }
//            else
//            {
//                tokenIds.Add(0); // O un valor para tokens desconocidos
//            }
//        }

//        return tokenIds;
//    }

//    static string DecodeTokens(float[] tokenIds, string tokenizerPath)
//    {
//        var tokenDict = LoadVocabulary(tokenizerPath);
//        var reverseDict = tokenDict.ToDictionary(pair => pair.Value, pair => pair.Key); // Invertir el diccionario
//        var decodedText = "";

//        foreach (var tokenId in tokenIds)
//        {
//            if (reverseDict.TryGetValue((int)tokenId, out string token))
//            {
//                decodedText += token + " ";
//            }
//        }
//        return decodedText.Trim();
//    }

//    static Dictionary<string, int> LoadVocabulary(string filePath)
//    {
//        var vocab = new Dictionary<string, int>();

//        using (var reader = new StreamReader(filePath))
//        {
//            string line;
//            while ((line = reader.ReadLine()) != null)
//            {
//                var parts = line.Split(',');
//                if (parts.Length == 2 && int.TryParse(parts[1], out int index))
//                {
//                    vocab[parts[0]] = index;
//                }
//            }
//        }

//        return vocab;
//    }
//}

//public class InputData
//{
//    [VectorType(8)] // Cambiado a 512 para coincidir con el tamaño de entrada
//    [ColumnName("input")]
//    public float[] Input { get; set; }
//}

//public class OutputData
//{
//    [ColumnName("output")]
//    public float[] Output { get; set; }
//}
