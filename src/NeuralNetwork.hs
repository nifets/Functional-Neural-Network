
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module NeuralNetwork (NeuralNetwork, NeuronLayer(Layer), createStd, createStd2, eval, train, trainVerbose, trainEpochsVerbose, cost, vector, costGradient, costs, minitrain, layers, feedForward, backPropagation, weights, dot, accuracy)
where


{- Some linear algebra stuff -}

import Data.Matrix
import Data.List (maximumBy)
import Data.Ord (comparing)

import Data.Random.Normal
import System.Random
import System.Random.Shuffle
import Control.Monad (foldM)

type R = Double

type Vector a = Matrix a -- column vector
{- I have some issue with the above definition, because it does not give the compiler any way to type check if an object with
type Vector is actually a column vector and not just an arbitrary matrix, but this makes writing the code nicer... -}

vector :: [R] -> Vector R --convert list to column vector
vector xs = fromList (length xs) 1 xs

(⊙) :: Vector R -> Vector R -> Vector R --Hadamard product (vector-vector element-wise multiplication)
v ⊙ u = elementwiseUnsafe (*) v u

(·) :: Vector R -> Vector R -> R -- dot product
v · u = sum (v ⊙ u)
dot = (·)

magnitude :: Vector R -> R
magnitude v = sqrt (v · v)

(⊗) :: Vector R -> Vector R -> Matrix R -- outer product
v ⊗ u = v * (transpose u)

scale = scaleMatrix



{-The logistic function, used to restrict the output of a sigmoid neuron to a value in the interval [0,1]. This is a smooth
alternative to the step function used by perceptrons. Hence, the output of a sigmoid neuron can be seen as the extent to which it
is activated, or rather the certainty of some decision that the neuron is supposed to make. -}
logistic :: R -> R
logistic x = 1.0 / (1.0 + exp (-x))

logistic' :: R -> R
logistic' x = logistic x * (1.0 - logistic x)



mse :: Vector R -> Vector R -> R -- mean squared error for single input
mse yv av = 0.5 * (v · v)
    where v = av - yv

dMse :: Vector R -> Vector R -> Vector R -- gradient of mean squared error for single input
dMse yv av = av - yv


crossEntropy :: Vector R -> Vector R -> R
crossEntropy yv av = -(yv · (fmap log av) + (fmap f yv) · (fmap (log.f) av))
    where f x = 1 - x

dCrossEntropy :: Vector R -> Vector R -> Vector R
dCrossEntropy yv av = elementwiseUnsafe (\y a -> y / max a eps - (1-y) / max (1-a) eps) yv av
    where eps = 1e-12


data NeuronLayer = Layer {weights :: Matrix R, -- weights matrix wm
                          biases  :: Vector R} -- biases vector bv
                          deriving (Show)

data NeuralNetwork = Network { layers :: [NeuronLayer],
                               objective :: Vector R -> Vector R -> R, -- cost function for one input
                               dObjective :: Vector R -> Vector R -> Vector R, -- gradient of the cost function
                               activation :: Vector R -> Vector R,  -- activation function σ applied to whole layer output
                               activation' :: Vector R -> Vector R, -- derivative of activation function σ' applied on vect.
                               rate :: R} -- learning rate η

instance Show NeuralNetwork where
    show (Network ls _ _ _ _ _) =
        foldr (\((Layer wm bv),n) txt -> "layer " ++ show n ++ ":\n" ++ show wm ++ "\n" ++ show bv ++ "\n\n" ++ txt) "" (zip ls [2..])

{- Some notation:
    l - layer
    w - weight
    b - bias
    x - input
    y - output
    a - activation
    z - weighted input sum
    e - error (partial derivative of cost w.r.t. z)

    types:
    v - vector
    m - matrix
    s - list
    (i.e.: avs is a list of vectors of activation, wm is a matrix of weights, etc.)
-}


{- Generate a neural network with randomized weights and biases from a seed, using the given cost, activation and rate to learn-}
create' :: [Int] -> (Vector R -> Vector R -> R) -> (Vector R -> Vector R -> Vector R) -> (Vector R -> Vector R) -> (Vector R -> Vector R) -> R -> Int -> NeuralNetwork
create' xs cost dCost σ σ' η seed = Network ls cost dCost σ σ' η
    where ls = zipWith Layer wms bvs
          wms = zipWith scale stdevs (randomMatrices (zip (tail xs) xs) seed)
          stdevs = map (recip . sqrt . fromIntegral) (init xs) -- Xavier: 1/sqrt(fan-in) => constant variance per neuron.
          bvs = map (\n -> zero n 1) (tail xs) -- just initialise biases with 0

          randomMatrices :: [(Int,Int)] -> Int -> [Matrix R] -- creates random matrices of the given sizes from some seed
          randomMatrices ns = zipWith (\(r,c) xs -> fromList r c xs) ns . splitList' (map (\(r,c) -> r*c) ns) . randomList

          randomList :: Int -> [R] -- creates an infinite list of i.i.d. r.v. of distribution N(0,1) from some seed
          randomList = mkNormals' (0,1)

          splitList' :: [Int] -> [a] -> [[a]] -- e.g. splitList' [n1,n2,n3,n4] xs splits xs into 4 lists of size n1,n2,n3,n4
          splitList' ns xs = (reverse . snd . foldl (\(as,bss) n -> (drop n as, (take n as) : bss)) (xs, []) ) ns


{- Generate a neural network using the global number generator -}
create :: [Int] -> (Vector R -> Vector R -> R) -> (Vector R -> Vector R -> Vector R) -> (Vector R -> Vector R) -> (Vector R -> Vector R) -> R -> IO NeuralNetwork
create xs cost dCost σ σ' η = fmap (create' xs cost dCost σ σ' η) randomIO


{- Create a simple Neural Network using the logistic function for activation and mean squared error for cost,
with randomized weights and biases; The first parameter gives the size of each layer -}
createStd :: [Int] -> R -> IO NeuralNetwork
createStd xs η = create xs mse dMse (fmap logistic) (fmap logistic') η

createStd2 :: [Int] -> R -> IO NeuralNetwork
createStd2 xs η = create xs crossEntropy dCrossEntropy (fmap logistic) (fmap logistic') η

{- What the neural network thinks the output of the given input should be-}
eval' :: NeuralNetwork -> Vector R -> Vector R
eval' nn = last . snd . feedForward nn --can be slightly optimized

{- Same as eval', but uses lists instead of vectors-}
eval :: NeuralNetwork -> [R] -> [R]
eval nn = toList . eval' nn . vector

cost :: NeuralNetwork -> [(Vector R, Vector R)] -> R
cost nn tdata = 1.0 / (fromIntegral . length) tdata 
    * (sum . map (uncurry (objective nn) . mapf (eval' nn))) tdata
    where mapf f (a,b) = (b, f a)

costGradient :: NeuralNetwork -> [(Vector R, Vector R)] -> R
costGradient nn tdata = 1.0 / (fromIntegral . length) tdata 
    * (magnitude . sum . map (uncurry (dObjective nn) . mapf (eval' nn))) tdata
    where mapf f (a,b) = (b, f a)

{- Runs an input through the neural network, computing the weighted input sums and the activations for each
layer. output is ([z2,z3,..,zL],[a1,a2,..,aL]) where a1 is just the input data-}
feedForward :: NeuralNetwork -> Vector R -> ([Vector R], [Vector R])
feedForward nn xv = (mapf tail . unzip . scanl advance (xv, xv) . layers) nn
    where advance (zv0,av0) (Layer wm bv) = (zv1, av1)   -- run the input from last layer through current layer
              where zv1 = wm * av0 + bv                  -- weighted input sum
                    av1 = (activation nn) zv1            -- output of current layer
          mapf f (a,b) = (f a, b)                        -- apply f to z-vectors, leave a-vectors unchanged


{- Computes the gradient of the cost function restricted to a single input, given the wanted output and the
weighted input sums and activations of each layer after running some input through the NN-}
backPropagation :: NeuralNetwork -> Vector R -> ([Vector R], [Vector R]) -> [(Matrix R, Vector R)]
backPropagation (Network ls _cost dCost _σ σ' _η) yv (zvs, avs) =
    let evs = (scanr cons nil . zip (tail wms)) zvs                   -- get δ by backpropagating through nn
        in zip (zipWith (⊗) evs avs) evs                             -- get ∂C/∂w and ∂C/∂b from δ errors
            where nil = dCost yv (last avs) ⊙ σ' (last zvs)          -- δᴸ where L is number of layers
                  cons (wm, zv) ev = ((transpose wm) * ev) ⊙ σ' zv   -- δˡ for l <- reverse [2..L-1]
                  wms = map weights ls                                -- weight matrices of the nn


{- Apply stochastic gradient descent to a minibatch selected by the training function -}
minitrain :: NeuralNetwork -> [(Vector R, Vector R)] -> NeuralNetwork
minitrain nn tdata =
    let newls = (zipWith layerGDescent ls . sumMGradients . map mGradient) tdata   -- apply gradient descent on the minibatch
        in (Network newls cost dCost σ σ' η)
            where layerGDescent (Layer wm bv) (dWm,dBv) =                          -- apply gradient descent to a single layer
                      Layer (wm - scale (η/m) dWm) (bv - scale (η/m) dBv)
                  sumMGradients = foldr1 (zipWith (\(a,b) (c,d) -> (a+c,b+d)))     -- add the minigradients together
                  mGradient (xv,yv) = (backPropagation nn yv  . feedForward nn) xv -- minigradient for one input
                  m = (fromIntegral . length) tdata                                -- size of minibatch
                  (Network ls cost dCost σ σ' η) = nn

{- Train the neural network on a given batch of data by using stochastic gradient descent on its cost function -}
train' :: RandomGen g => NeuralNetwork -> Int -> [([R], [R])] -> g -> NeuralNetwork
train' nn batchSize tdata gen = (foldl minitrain nn . splitList batchSize . vectorized) (shuffle' tdata (length tdata) gen)
    where splitList n = takeWhile (not . null) . map (take n) . iterate (drop n) -- produces list of minibatches
          vectorized xs = map (\(a,b) -> (vector a, vector b)) xs                -- convert data from list to vector

train'' nn batchSize tdata gen = (map snd . scanl cons (nn,0) . splitList batchSize . vectorized) (shuffle' tdata (length tdata) gen)
    where splitList n = takeWhile (not . null) . map (take n) . iterate (drop n) -- produces list of minibatches
          vectorized xs = map (\(a,b) -> (vector a, vector b)) xs                -- convert data from list to vector
          cons (nnn, _) xs = let nnn' = minitrain nnn xs
                             in (nnn', costGradient nnn' xs)

train :: NeuralNetwork -> Int -> [([R],[R])] -> IO NeuralNetwork
train nn batchSize tdata = fmap (train' nn batchSize tdata) getStdGen

-- print costs after every minibatch
trainVerbose :: NeuralNetwork -> Int -> [([R],[R])] -> IO NeuralNetwork
trainVerbose nn batchSize tdata = do
    gen <- getStdGen
    let batches = (splitList batchSize . vectorized) (shuffle' tdata (length tdata) gen)
    foldM step nn (zip [(1 :: Int)..] batches)
    where step nnn (i, batch) = do
              let nnn' = minitrain nnn batch
              putStrLn $ "batch " ++ show i ++ "  cost: " ++ show (costGradient nnn' batch)
              return nnn'
          splitList n = takeWhile (not . null) . map (take n) . iterate (drop n)
          vectorized xs = map (\(a,b) -> (vector a, vector b)) xs

costs nn batchSize tdata = fmap (train'' nn batchSize tdata) getStdGen

-- train for n epochs on training data, printing test accuracy after each epoch --
trainEpochsVerbose :: Int -> NeuralNetwork -> Int -> [([R],[R])] -> [([R],[R])] -> IO NeuralNetwork
trainEpochsVerbose numEpochs nn batchSize trainData testData =
    foldM trainEpoch nn [1..numEpochs]
    where trainEpoch n epoch = do
              putStrLn $ "\n--- Epoch " ++ show epoch ++ " ---"
              trained <- trainVerbose n batchSize trainData
              let acc = accuracy trained testData
              putStrLn $ "Epoch " ++ show epoch ++ " accuracy: " ++ show acc
              return trained

accuracy nn tdata = (sum . map f) tdata / (fromIntegral (length tdata))
    where f (a,b) = if argmax (eval' nn (vector a)) == argmax (vector b) then 1.0 else 0.0

argmax :: Vector R -> Int
argmax v = fst $ maximumBy (comparing snd) (zip [0..] (toList v))
