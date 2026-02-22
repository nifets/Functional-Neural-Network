{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import NeuralNetwork
import Text.CSV
import Data.Either
import Data.Tuple

main :: IO ()
main = do
    rawTrainData <- (fmap (tail . map (map read) . fromRight []) . parseCSVFromFile) "data/mnist_train.csv" :: IO [[Double]]
    rawTestData <- (fmap (tail . map (map read) . fromRight []) . parseCSVFromFile) "data/mnist_test.csv" :: IO [[Double]]
    
    let processData = map ((\(pixels, label) -> (map (/255) pixels, label)) . fmap (\[a] -> take (round a) (repeat 0.0) ++ [1.0] ++ take (9 - round a) (repeat 0.0)) . swap . splitAt 1)
    let trainData = filter (\(p, _) -> length p == 784) $ processData rawTrainData
    let testData = filter (\(p, _) -> length p == 784) $ processData rawTestData
    
    nn <- createStd [784, 30, 10] 0.1
    trainedNN <- trainEpochsVerbose 20 nn 10 trainData testData

    putStrLn $ "\nFinal accuracy: " ++ show (accuracy trainedNN testData)
    let compare nn (a,b) = (eval nn a, b)
    print (map (compare trainedNN) (take 10 testData))
