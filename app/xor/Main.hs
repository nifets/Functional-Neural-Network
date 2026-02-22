module Main where

import Control.Monad (foldM)
import NeuralNetwork

xorData :: [([Double], [Double])]
xorData =
    [ ([0, 0], [1, 0])
    , ([0, 1], [0, 1])
    , ([1, 0], [0, 1])
    , ([1, 1], [1, 0])
    ]

main :: IO ()
main = do
    nn <- createStd [2, 4, 2] 0.5
    trainedNN <- foldM (\n _ -> train n 4 xorData) nn [1..5000 :: Int]
    putStrLn "XOR predictions:"
    mapM_ (printPrediction trainedNN) xorData
    putStrLn $ "accuracy: " ++ show (accuracy trainedNN xorData)

printPrediction :: NeuralNetwork -> ([Double], [Double]) -> IO ()
printPrediction nn (input, expected) =
    putStrLn $ show input ++ " -> predicted: " ++ show predicted ++ "  expected: " ++ show (round (last expected) :: Int)
    where predicted = eval nn input
