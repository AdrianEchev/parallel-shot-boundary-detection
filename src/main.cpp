/*
 * LEAD University
 * BCD-9218 // Computación Paralela y Distribuida
 * Proyecto Final
 * Adrián Echeverría
 * Lolita Maldonado
 */

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include "timer.h"

// mpic++ -fopenmp clean.cpp -o clean `pkg-config --cflags --libs opencv4`
// export OMP_NUM_THREADS=2 ; mpirun -np 2 ./clean short.mp4 data

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // INITIALIZE VARIABLES
    int size, rank, half_i, half_frames, total_frames, count, TAG = 888;
    double elapsed_time, correlation;
    string arg1, path, filename, text;
    Mat frame, hsv, image1, image2, hist1, hist2;
    bool success;
    const char *cuts;

    // INITIALIZE HISTOGRAM VARIABLES
    int h_bins = 50, s_bins = 32, v_bins = 10;
    int histSize[] = {h_bins, s_bins, v_bins};
    float h_ranges[] = {0, 180}, s_ranges[] = {0, 256}, v_ranges[] = {0, 256};
    const float *ranges[] = {h_ranges, s_ranges, v_ranges};
    int channels[] = {0, 1, 2};

    // INITIALIZE MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // START TIMER
    timerStart();

    // IF THERE IS INPUT AND 1 RANK
    if (argc > 2 && size == 1)
    {
        // GENERATE INITIAL VALUES
        string arg2(argv[2]);
        text = arg2 + ".txt";
        ofstream file(text);

        string arg1(argv[1]);
        path = "./video_files_here/" + arg1;
        VideoCapture cap(path);
        if (!cap.isOpened())
        {
            printf("Error: video not found in '/video_files_here/'. \n");
        }

        total_frames = cap.get(CAP_PROP_FRAME_COUNT);
        double corr[total_frames];

        if (rank == 0)
        {
            for (int i = 1; i < total_frames; i++)
            {
                success = cap.read(frame);
                if (!success)
                {
                    printf("Error: frame not found in video. \n");
                }

                // RBG TO HSV
                cvtColor(frame, frame, COLOR_BGR2HSV);

                // STORE IN FOLDER
                filename = to_string(i) + ".png";
                imwrite("./frames_output/" + filename, frame);
            }

            for (int k = 1; k < total_frames - 1; k++)
            {
                count = k + 1;

                // TWO PICTURES FOR TWO HISTOGRAMS FOR ONE CORRELATION
                image1 = imread("./frames_output/" + to_string(k) + ".png");
                image2 = imread("./frames_output/" + to_string(count) + ".png");

                calcHist(&image1, 1, channels, Mat(), hist1, 3, histSize, ranges, true, false);
                calcHist(&image2, 1, channels, Mat(), hist2, 3, histSize, ranges, true, false);

                normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
                normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

                correlation = compareHist(hist1, hist2, 0);

                // WRITE IN .TXT FILE IF THERE ARE LOW CORRELATIONS
                if (correlation < 0.5)
                {
                    text = "Low correlation found: " + to_string(correlation) + ". Discrepancy can be found between frames " + to_string(k) + " and " + to_string(k + 1) + " \n";
                    cuts = text.c_str();
                    file << cuts;
                }
            }
            // STOP TIMER
            elapsed_time = timerStop();
            text = "\nExecution time: " + to_string(elapsed_time) + " seconds.\nMPI Ranks: " + to_string(size) + ". Total frames processed: " + to_string(total_frames - 1);

            const char *timer = text.c_str();
            file << timer;
            file.close();
            // END
        }
    }

    // IF THERE IS INPUT AND 2 RANKS
    else if (argc > 2 && size == 2)
    {
        // GENERATE INITIAL VALUES
        string arg2(argv[2]);
        text = arg2 + ".txt";
        ofstream file(text);

        string arg1(argv[1]);
        path = "./video_files_here/" + arg1;
        VideoCapture cap(path);
        if (!cap.isOpened())
        {
            printf("Error: video not found in '/video_files_here/'. \n");
        }

        total_frames = cap.get(CAP_PROP_FRAME_COUNT);
        double corr[total_frames];

        // RANK 1 IS TAKING THE SECOND HALF OF FRAMES
        if (rank == 1)
        {
            for (int i = (total_frames / 2); i < (total_frames / 2); i++)
            {
                success = cap.read(frame);
                if (!success)
                {
                    printf("Error: frame not found in video. \n");
                }

                cvtColor(frame, frame, COLOR_BGR2HSV);

                filename = to_string(i) + ".png";
                imwrite("./frames_output/" + filename, frame);
            }
        }

        // RANK ROOT IS TAKING THE FIRST HALF OF FRAMES
        if (rank == 0)
        {
            for (int i = 1; i < (total_frames / 2); i++)
            {
                success = cap.read(frame);
                if (!success)
                {
                    printf("Error: frame not found in video. \n");
                }

                cvtColor(frame, frame, COLOR_BGR2HSV);

                filename = to_string(i) + ".png";
                imwrite("./frames_output/" + filename, frame);
            }

            for (int k = 1; k < total_frames - 1; k++)
            {
                count = k + 1;

                image1 = imread("./frames_output/" + to_string(k) + ".png");
                image2 = imread("./frames_output/" + to_string(count) + ".png");

                calcHist(&image1, 1, channels, Mat(), hist1, 3, histSize, ranges, true, false);
                calcHist(&image2, 1, channels, Mat(), hist2, 3, histSize, ranges, true, false);

                normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
                normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

                correlation = compareHist(hist1, hist2, 0);
                corr[k - 1] = correlation;
            }
            // SMALL OMP PARALLELIZATION
#pragma omp parallel for
            for (int k = 1; k < total_frames - 1; k++)
            {
                if (corr[k - 1] < 0.5)
                {
                    text = "Low correlation found: " + to_string(corr[k - 1]) + ". Discrepancy can be found between frames " + to_string(k) + " and " + to_string(k + 1) + " \n";
                    cuts = text.c_str();
                    file << cuts;
                }
            }
            elapsed_time = timerStop();
            text = "\nExecution time: " + to_string(elapsed_time) + " seconds.\nMPI Ranks: " + to_string(size) + ". Total frames processed: " + to_string(total_frames - 1);

            const char *timer = text.c_str();
            file << timer;
            file.close();
            // END
        }
    }
    MPI_Finalize();
    return 0;
    // MAIN END
}
// PROGRAM FINISHED
