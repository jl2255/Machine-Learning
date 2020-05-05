import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoring {
    /** The n value */
    private static final int N = 100; // number of vertices
    private static final int L =5; // L adjacent nodes per vertex
    private static final int K = 10; // K possible colors
    /**
     * The test main
     * @param args ignored
     */

    public static void main(String[] args) {
    	
    	double start, end, time;

	int iter = 5000;
    double sum_rhc = 0;
    double sum_sa = 0;
    double sum_ga = 0;
    double sum_mimic = 0;

    double time_rhc = 0;
    double time_sa = 0;
    double time_ga = 0;
    double time_mimic = 0;
    
    for (int n = 0; n < 10; n++) {
        Random random = new Random(N*L);
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;	
            vertex.setAdjMatrixSize(L);
            for(int j = 0; j <L; j++ ){
            	 vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
            }
        }
       
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        Distribution df = new DiscreteDependencyTree(.1); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        start = System.nanoTime();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
        fit.train();
        end = System.nanoTime();
        time = end - start;
        time /= Math.pow(10, 9);
        sum_rhc += ef.value(rhc.getOptimal());
        time_rhc += time;

        start = System.nanoTime();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .9, hcp);
        fit = new FixedIterationTrainer(sa, iter);
        fit.train();
        end = System.nanoTime();
        time = end - start;
        time /= Math.pow(10, 9);
        sum_sa += ef.value(sa.getOptimal());
        time_sa += time;
        
        start = System.nanoTime();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 250, 50, gap);
        fit = new FixedIterationTrainer(ga, iter);
        fit.train();
        end = System.nanoTime();
        time = end - start;
        time /= Math.pow(10, 9);
        sum_ga += ef.value(ga.getOptimal());
        time_ga += time;
        
        start = System.nanoTime();
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, iter);
        fit.train();
        end = System.nanoTime();
        time = end - start;
        time /= Math.pow(10, 9);
        sum_mimic += ef.value(mimic.getOptimal());
        time_mimic += time;
    }
    double average_rhc = sum_rhc / 10;
    double average_sa = sum_sa / 10;
    double average_ga = sum_ga / 10;
    double average_mimic = sum_mimic / 10;

    double averagetime_rhc = time_rhc / 10;
    double averagetime_sa = time_sa / 10;
    double averagetime_ga = time_ga / 10;
    double averagetime_mimic = time_mimic / 10;
    
    System.out.println("##############");
    System.out.println("this is iteration " + iter);
    System.out.println("rhc average is " + average_rhc + ", time average is " + averagetime_rhc);
    System.out.println("sa average is " + average_sa + ", time average is " + averagetime_sa);
    System.out.println("ga average is " + average_ga + ", time average is " + averagetime_ga);
    System.out.println("mimic average is " + average_mimic + ", time average is " + averagetime_mimic);
    
    }
}
