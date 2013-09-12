/* This is Scott Fahlman's quickprop program translated from Common Lisp
 * into C by Terry Regier at the University of California, Berkeley.
 * Netmail address: regier@cogsci.berkeley.edu
 * This version is Quickprop 1 from September, 1988.
 *
 * An example of network setup data is included at the end of this file.
 * 
 * The algorithm and some test results are described in Fahlman's paper
 * "Faster-Learning Variations on Back-Propagation: An Empirical Study"
 * in Proceedings of 1988 Connectionist Models Summer School, published
 * by Morgan Kaufmann.
 *
 * Note: the parameter called "mu" in the paper is called "max-factor" here.
 * 
 * Changes made to quickprop.c version 1 by N Karunanithi netmail:
 * <karunani@handel.cs.colostate.edu>.
 * 
 * Connections can now be specified for multiple ranges of units.
 * For example if you had 3 layers of hidden units and wanted the
 * third layer to have connections to inputs and the second layer,
 * but not the first hidden layer.
 * 
 * Bug fix in CONNECT_LAYERS by Richard Dale Romero <rr2p+@andrew.cmu.edu>
 * inserted into the code on September 18, 1991
 *
 * You may specify hidden and output units as sigmoids with ranges
 * of -0.5 to 0.5 (SIGMOIDAL) or from 0.0 to 1.0 (ASYMSIGMOIDAL) in
 * the input file.
 */

#include <stdio.h>
#include <math.h>

#define N 30              /* Max number of units allowed in net */

#define SIGMOID        1  /* Unit_type is SIGMOID with output = +0.5 to -0.5 */
#define ASYMSIGMOID    2  /* ASYMSIGMOID with output = 0.0 to 1.0 */


/*  Global variables */

int   Epoch;               /* Current epoch number */
float WeightRange;         /* Random-init weights in range [-WR,+WR] */
float SigmoidPrimeOffset;  /* Add to sigmoid-prime to kill flat spots */
int   HyperErr;            /* 1 => use atanh error function */
int   SplitEpsilon;        /* 1 => divide epsilon by fan-in before use */
float Epsilon;             /* For grad descent if last step was (almost) 0 */
float Momentum;            /* Normal old momentum term */
float ModeSwitchThreshold; /* Inside thresh, do grad descent; outside, jump. */
float MaxFactor;           /* Don't jump more than this times last step */
float Decay;               /* Weight decay */
int   SinglePass;          /* 1 => Pause after forward/backward cycle */
int   SingleEpoch;         /* 1 => Pause after each training epoch */
int   Step;                /* Turned to 1 after each pause, briefly */
int   Restart;             /* 1 => restart when max epochs reached */
int   KeepScore;           /* 1 => accumulate error score for each epoch */
float TotalError;          /* Total output error for one epoch */
float ScoreThreshold;      /* This close to desired value => bit is correct */
int   TotalErrorBits;      /* Total # bits in epoch that were wrong */
int   DidGradient;         /* Total # patterns that did gradient descent */

int   Nunits;              /* Total number of units in net */
int   Ninputs;             /* Number of input units */
int   FirstHidden;         /* Index of 1st hidden unit */
int   Nhidden;             /* Number of hidden units */
int   FirstOutput;         /* Index of 1st output unit */
int   Noutputs;            /* Number of output units */
int   Unit_type;           /* Type of hidden and Output Units: 1=> SIGMOID,and
                              2 => ASYMSIGMOID */

float Outputs[N];          /* Final output value for each unit */
float ErrorSums[N];        /* Total error activation for each unit */
float Errors[N];           /* Final error value for each unit */
int   Nconnections[N];     /* # of INCOMING connections per unit */
int   Connections[N][N];   /* C[i][j] lists jth unit projecting to unit i */
float Weights[N][N];       /* W[i][j] holds weight of C[i][j] */
float DeltaWeights[N][N];  /* Change between previous weight and current one */
float Slopes[N][N];        /* Accumulated slope value for each position */
float PrevSlopes[N][N];    /* Similarly, for the last position visited */

int   NTrainingPatterns;        /* !! Not in Lisp version.  Needed here. */
int   NTestPatterns;            /* !! Not in Lisp version.  Needed here. */
float TrainingInputs[200][N];
float TrainingOutputs[200][N];
float TestInputs[200][N];
float TestOutputs[200][N];

float tinputs[N];           /* Input vector to be tested. */


main ()
{

    long seed;
    long lrand48();

    int i, j, epox, response;
    float RANDOM_WEIGHT();
    char fname[80];


    /* Start up the random number generator */
    printf ("Enter seed for random number generator:  ");
    scanf ("%d", &seed); 
    srand(time(0));

    INITIALIZE_GLOBALS();

    /* Get network */
    printf ("Enter name of network: ");
    scanf ("%s", fname);

    GET_NETWORK_CONFIGURATION(fname);
    printf ("Want to retrieve old weights [0=>no, 1=>yes]? ");
    scanf ("%d", &response);
    if ( response )
        GET_WEIGHTS(fname);

    /* Train the sucker. */
    epox = 34;
    while ( epox != 0 )
    {
        printf ("Enter number of epochs to train: ");
        scanf ("%d", &epox);
        if ( epox != 0 )
            TRAIN ( epox );
    }

    /* Test the sucker. */
    printf ("Want to test [0=>no, 1=>yes]? ");
    scanf ("%d", &response);
    if ( response != 0 )
        TEST();

    printf ("Want to dump weights [0=>no, 1=>yes]? ");
    scanf ("%d", &response);
    if ( response )
        DUMP_WEIGHTS ( fname );
}


/*
 *  Get and initialize a network. 
 */
GET_NETWORK_CONFIGURATION(fname)
    char fname[];
{
    FILE  *infile, *fopen();
    char junk[5][80];
    char stringjunk[80];
    char realfname[80];
    char c;
    int  temp[10], i, j, connect;

    sprintf (realfname, "%s.net", fname);
    infile = fopen ( realfname, "r" );

    c = 'c';           /* Discard leading comments */
    //while (c != '#') 
    //  fscanf (infile, "%c", &c);

    /* Get numbers of inputs, hidden units, and output units */
    fscanf (infile, "%s %d %s %d %s %d", 
            junk[0], &temp[0], junk[1], &temp[1], junk[2], &temp[2]);
    BUILD_DATA_STRUCTURES( temp[0], temp[1], temp[2] );

    /* Get the type units used in hidden and outpt layers. */
    fscanf (infile, "%s %d ", junk[0], &temp[0]);
    if (temp[0] == 1) 
        Unit_type = SIGMOID;
    else if (temp[0] == 2) 
        Unit_type = ASYMSIGMOID;

    /* Connect layers. */
    fscanf (infile, "%s %d", junk[0], &connect);

    for (i=0; i<connect; i++)          /* Reading CONNECT_LAYERS lines */
    {
        fscanf (infile, "%d %d %d %d",
                &temp[0], &temp[1], &temp[2], &temp[3]);
        CONNECT_LAYERS ( temp[0], temp[1], temp[2], temp[3] );
    }

    /* Read in number of training patterns, then patterns themselves */
    fscanf (infile, "%s %d", junk[0], &NTrainingPatterns);
    for (i=0; i<NTrainingPatterns; i++)
    {
        for (j=0; j<Ninputs; j++)
            fscanf (infile, "%f", &TrainingInputs[i][j]);
        for (j=0; j<Noutputs; j++)
            fscanf (infile, "%f", &TrainingOutputs[i][j]);
    }

    /* Read in number of test patterns, then patterns themselves */
    fscanf (infile, "%s %d", junk[0], &NTestPatterns);
    for (i=0; i<NTestPatterns; i++)
    {
        for (j=0; j<Ninputs; j++)
            fscanf (infile, "%f", &TestInputs[i][j]);
        for (j=0; j<Noutputs; j++)
            fscanf (infile, "%f", &TestOutputs[i][j]);
    }

    fclose(infile);
}


/*
 *  Dump weights in the specified file.
 */
DUMP_WEIGHTS(fname)
    char fname[];
{
    FILE  *outfile, *fopen();
    int  i, j;
    char realfname[80];

    /* Dump weights */
    sprintf (realfname, "%s.wts", fname);
    outfile = fopen ( realfname, "w" );
    for (i=0; i<N; i++)
        for (j=0; j<N; j++)
            if ( Weights[i][j] != 0.0 )
                fprintf (outfile, "%d %d %f ", i, j, Weights[i][j]);

    fprintf (outfile, "%d %d %f ", -1, -1, -1.0);   /* Signal EOF */

    fclose (outfile);
}


/*
 *  Get weights from the specified file.
 */
GET_WEIGHTS(fname)
    char fname[];
{
    FILE  *infile, *fopen();
    int  i, j;
    float inweight;
    char realfname[80];

    /* Get weights */
    sprintf (realfname, "%s.wts", fname);
    infile = fopen ( realfname, "r" );
    for (i=0; i<N; i++)
        for (j=0; j<N; j++)
            Weights[i][j] = 0.0;         /* Default weight */

    i = 11;                          /* Arbitrary +ive */
    while ( i >= 0 )
    {
        fscanf (infile, "%d %d %f", &i, &j, &inweight);
        if ( i >= 0 )
            Weights[i][j] = inweight;
    }

    fclose (infile);
}


INITIALIZE_GLOBALS()
{
    Unit_type = SIGMOID;
    Epoch = 0;
    WeightRange = 0.7;
    SigmoidPrimeOffset = 0.1;
    HyperErr = 1;
    SplitEpsilon = 1;
    Epsilon = 0.55; /* 1.0 */
    Momentum = 0.9; /* 0.0 */
    ModeSwitchThreshold = 0.0;
    MaxFactor = 1.75; /* 1.75 */
    Decay = -0.0001; /* -0.0001 */
    SinglePass = SingleEpoch = 0;
    Step = KeepScore = 0;
    Restart = 1;
    TotalError = 0.0;
    ScoreThreshold = 0.35;
    TotalErrorBits = 0;
}


BUILD_DATA_STRUCTURES (ninputs, nhidden, noutputs)
    int	ninputs, nhidden, noutputs;
{
    int i;

    Nunits      = 1 + ninputs + nhidden + noutputs;
    Ninputs     = ninputs;
    FirstHidden = 1 + ninputs;
    Nhidden     = nhidden;
    FirstOutput = 1 + ninputs + nhidden;
    Noutputs    = noutputs;

    for (i=0; i<=Nunits; i++)    Outputs[i] = 0.0;
    for (i=0; i<=Nunits; i++)    ErrorSums[i] = 0.0;
    for (i=0; i<=Nunits; i++)    Errors[i] = 0.0;
    for (i=0; i<=Nunits; i++)    Nconnections[i] = 0;

    Outputs[0] = 1.0;        /* The bias unit */
}


/*
 * Return a float between -range and +range.
 */
float RANDOM_WEIGHT (range)
    float range;
{
    return ( (float) (range * (rand()%1000 / 500.0)) - range );
}


/*
 *  Build a connection from every unit in range1 to every unit in range2.
 *  Also add a connection from the bias unit (unit 0) to every unit in 
 *  range2.  Set up random weights on links.
 */
CONNECT_LAYERS (start1, end1, start2, end2)
    int	start1, end1, start2, end2;
{

    int n, i, j, k;

    Epoch = 0;

    for (i=start2; i<=end2; i++)
    {
        if(Nconnections[i] == 0){
            Nconnections[i]  += 1;
            Connections[i][0] = 0;
            Weights[i][0] = RANDOM_WEIGHT(WeightRange);
            DeltaWeights[i][0] = 0.0;
            Slopes[i][0] = 0.0;
            PrevSlopes[i][0] = 0.0;
            k = 1;
        }
        else 
            k = Nconnections[i]; 
        /*	k =  start1;           Bug found by 
            Richard Dale Romero <rr2p+@andrew.cmu.edu> */

        for (j=start1; j<=end1; j++){
            Nconnections[i]  += 1;
            Connections[i][k] = j;
            Weights[i][k] = RANDOM_WEIGHT(WeightRange);
            DeltaWeights[i][k] = 0.0;
            Slopes[i][k] = 0.0;
            PrevSlopes[i][k] = 0.0;
            k++;
        }
    }
}


/* 
 *  For each connection, select a random initial weight between WeightRange
 *  and its negative.  Clear delta and previous delta values.
 */
INIT_WEIGHTS()
{
    int i, j;

    for (i=0; i<Nunits; i++)
        for (j=0; j<Nconnections[i]; j++)
        {
            Weights[i][j] = RANDOM_WEIGHT(WeightRange);
            DeltaWeights[i][j] = 0.0;
            Slopes[i][j] = 0.0;
            PrevSlopes[i][j] = 0.0;
        }
}


/*
 *  Save the current slope values as PrevSlopes, and "clear" all current
 *  slopes (actually set to corresponding weight, decayed a bit).
 */
CLEAR_SLOPES()
{
    int i, j;

    for (i=FirstHidden; i<Nunits; i++)
        for (j=0; j<Nconnections[i]; j++)
        {
            PrevSlopes[i][j] = Slopes[i][j];
            Slopes[i][j] = ( Decay * Weights[i][j] );
        }
}


/*
 * Given the sum of weighted inputs, compute the unit's activation value.
 * Defined unit types are SIGMOID and ASYMSIGMOID.
 */
float ACTIVATION(sum)
    float sum;
{

    switch(Unit_type){
        case SIGMOID: 
            /* Symmetrical sigmoid function in range -0.5 to +0.5. */
            if (sum < -15.0) 
                return(-0.5);
            else if (sum > 15.0) 
                return(0.5);
            else 
                return (1.0 /(1.0 + exp(-sum)) - 0.5);
        case ASYMSIGMOID: 
            /* asymmetrical sigmoid function in range 0.0 to 1.0. */
            if (sum < -15.0) 
                return(0.0);
            else if (sum > 15.0) 
                return(1.0);
            else 
                return (1.0 /(1.0 + exp(-sum)));
    }
}

/*
 * Given the unit's activation value and sum of weighted inputs, compute
 * the derivative of the activation with respect to the sum.  Defined unit
 * types are SIGMOID and ASYMSIGMOID.
 */
float ACTIVATION_PRIME(value)
    float value;
{
    switch(Unit_type){
        case SIGMOID: 
            /* Symmetrical sigmoid function. */
            return (SigmoidPrimeOffset + (0.25 -  value*value));
        case ASYMSIGMOID: 
            /* asymmetrical sigmoid function in range 0.0 to 1.0. */
            return (SigmoidPrimeOffset + (value * (1.0 - value)));
    }
}

/*
 *  Compute the error for one output unit.  
 *  HyperErr==0 => use squared error.
 *  HyperErr==1 => use atanh.
 */
float ERRFUN (desired, actual)
    float desired, actual;
{
    float dif;

    dif = desired - actual;

    if ( KeepScore )   
    {
        TotalError += dif*dif;
        if ( fabs(dif) >= ScoreThreshold )
            TotalErrorBits++;
    }

    if ( HyperErr == 0 )         /* Not using atanh for error */
    {
        if ((-0.1 < dif) && (dif < 0.1))
            return (0.0);
        else 
            return (dif);
    }
    else                         /* Using atanh for error */
    {
        if ( dif < -.9999999 )
            return (-17.0);
        else if ( dif > .9999999 )
            return (17.0);
        else
            return ( log ( (1.0+dif) / (1.0-dif) ) );
    }
}


/*
 *  This is it, ya Habaayib:  the forward pass in BP.
 */
FORWARD_PASS (input)
    float input[];
{
    int i, j;
    float sum;

    /* Load in the input vector */
    for (i=0; i<Ninputs; i++)
        Outputs[i+1] = input[i];

    /* For each unit, collect incoming activation and pass through sigmoid. */
    for (j=FirstHidden; j<Nunits; j++)
    {
        sum = 0.0;
        for (i=0; i<Nconnections[j]; i++)
            sum += ( Outputs[ Connections[j][i] ] * Weights[j][i] );
        Outputs[j] = ACTIVATION(sum);
    }
}


/*
 *  Goal is a vector of desired values for the output units.  Propagate
 *  the error back through the net, accumulating weight deltas.
 */
BACKWARD_PASS (goal)
    float	goal[];
{
    int i, j, cix;     /* cix is "connection index" */

    /* Compute error sums for output and other nodes */
    for (i=FirstOutput; i<Nunits; i++)     /*  !! should it really be "<"? */
        ErrorSums[i] = ERRFUN( goal[i-FirstOutput], Outputs[i]);
    for (i=0; i<FirstOutput; i++)
        ErrorSums[i] = 0.0;

    /* Back-prop.  When we reach a given unit in loop, error from all later
     * units will have been collected.
     */
    for (j=Nunits-1; j>=FirstHidden; j--)
    {
        Errors[j] = ACTIVATION_PRIME(Outputs[j]) * ErrorSums[j];
        for (i=0; i<Nconnections[j]; i++)
        {
            cix = Connections[j][i];
            ErrorSums[cix] += ( Errors[j] * Weights[j][i] );
            Slopes[j][i] += ( Errors[j] * Outputs[cix] );
        }
    }
}


/*
 *  Update all weights in the network as a function of each weight's current
 *  slope, previous slope, and the size of the last jump.
 */
UPDATE_WEIGHTS()
{
    int i, j;
    float next_step, shrink_factor;

    shrink_factor = MaxFactor / ( 1.0 + MaxFactor );

    for (j=FirstHidden; j<Nunits; j++)
        for (i=0; i<Nconnections[j]; i++)
        {
            next_step = 0.0;

            if ( DeltaWeights[j][i] > ModeSwitchThreshold )
            {                            /* Last step was signif. +ive..... */
                if ( Slopes[j][i] > 0.0 )  /* Add in epsilon if +ive slope */
                    next_step += (SplitEpsilon ? 
                            ( (Epsilon * Slopes[j][i]) / Nconnections[j] ) :
                            ( Epsilon * Slopes[j][i] ));
                /* If slope > (or close to) prev slope, take max size step. */
                if ( Slopes[j][i] > (shrink_factor * PrevSlopes[j][i]) )
                    next_step += ( MaxFactor * DeltaWeights[j][i] );
                else        /*  Use quadratic estimate */
                    next_step += ( (Slopes[j][i]/(PrevSlopes[j][i]-Slopes[j][i])) 
                            * DeltaWeights[j][i] );
            }
            else if ( DeltaWeights[j][i] < -ModeSwitchThreshold )
            {                          /* Last step was signif. -ive.... */
                if ( Slopes[j][i] < 0.0 )/* Add in epsilon if -ive slope */
                    next_step += (SplitEpsilon ? 
                            ( (Epsilon * Slopes[j][i]) / Nconnections[j] ) :
                            ( Epsilon * Slopes[j][i] ));
                /* If slope < (or close to) prev slope, take max size step. */
                if ( Slopes[j][i] < (shrink_factor * PrevSlopes[j][i]) )
                    next_step += ( MaxFactor * DeltaWeights[j][i] );
                else        /*  Use quadratic estimate */
                    next_step += ( (Slopes[j][i]/(PrevSlopes[j][i]-Slopes[j][i])) 
                            * DeltaWeights[j][i] );
            }
            else       /* Normal gradient descent, complete with momentum */
            {
                DidGradient++;
                next_step += ((SplitEpsilon ? 
                            ( (Epsilon * Slopes[j][i]) / Nconnections[j] ) :
                            ( Epsilon * Slopes[j][i] ))
                        + (Momentum * DeltaWeights[j][i]) );
            }

            /* Set delta weight, and adjust the weight itself. */
            DeltaWeights[j][i] = next_step;
            Weights[j][i] += next_step;
        }
}

/*
 *  Perform forward and back propagation once for each pattern in the
 *  training set, collecting deltas.  Then burn in the weights.
 */
TRAIN_ONE_EPOCH()
{
    int i;

    CLEAR_SLOPES();

    for (i=0; i<NTrainingPatterns; i++)
    {
        FORWARD_PASS ( TrainingInputs[i] );
        BACKWARD_PASS ( TrainingOutputs[i] );
    }

    UPDATE_WEIGHTS();
    Epoch++;
}


/*
 *  Train the network for the specified number of epochs, printing out
 *  performance stats every 10 epochs.
 */
TRAIN ( times )
    int times;
{
    int i, report;

    report = 50;

    for (i=0; i<times; i++)
    {
        if ( Epoch % report == 0 )     /* Time to report status */
        {
            DidGradient = 0;
            KeepScore = 1;
            TotalError = 0.0;
            TotalErrorBits = 0;
            TRAIN_ONE_EPOCH();
            printf ("Epoch %d:  %d Bits Wrong, Total Error = %f, DidGradient = %d.\n",
                    (Epoch - 1), TotalErrorBits, TotalError, DidGradient);
            KeepScore = 0;
        }
        else
            TRAIN_ONE_EPOCH();	
    }
}


TEST ()
{
    int i,j;

    tinputs[0] = 1.0;          /* Initial nonzero value */

    while (tinputs[0] >= 0.0)
    {
        /* printf ("Enter the %d input values [first one less than 0.0 => quit]: ", 
           Ninputs);
           for (i=0; i<Ninputs; i++)
           scanf ("%f", &tinputs[i]);
           */
        for(j=0; j<NTestPatterns; j++){
            for (i=0; i<Ninputs; i++)
                tinputs[i] = TestInputs[j][i];
            FORWARD_PASS(tinputs);
            printf ("Output for test pattern %d is: ",j);
            for (i=0; i<Noutputs; i++)
                printf ("%f  ", Outputs[FirstOutput+i]);
            printf ("\n");
        }
        tinputs[0] = -0.5;
    }
}
/*--------------------- cut here ----------------------------------------*/
/* 3 example net files follow.  2 versions of XOR and one larger problem */
/*-------------- This is  an example network with shortcut connections --*/
/*  This is network of 2=2=1 architecture with links between the input layer
 **  and the output layer. In this case the output node receives input from
 **  all other nodes in the network.
 **
 **  The third line specifies the unit type. Possible unit types are:
 **  1 for SIGMOIDAL and 2 for ASYMSIGMOIDAL.
 */
/*------------  End of comment   ---------------------------------------*/
/* Network specification begins from the line below the current one. */
#ifdef YOU_WANT_TO_COMPILE_JUNK
XOR with 1 hidden unit and crosscut connections.
1st Connect call connects inputs to both hidden and output, 2nd call connects
hidden unit to the output unit.
#
/* start of net
   Ninputs 2          Nhidden 1          Noutputs 1
   UnitType  1
   Connectcalls 2
   1 2    3 4
   3 3    4 4
   NTrainingPatterns 4
   0.0  0.0 	 -0.5
   0.0  1.0 	 0.5
   1.0  0.0 	 0.5
   1.0  1.0 	 -0.5 
   NTestPatterns 4
   0.0  0.0 	 -0.5
   0.0  1.0 	 0.5
   1.0  0.0 	 0.5
   1.0  1.0 	 -0.5 
   end of first example */
/*-------------- This is  an example network with shortcut connections --*/
/*  This is network of 2=2=1 architecture with links between the input layer
 **  and the output layer. In this case the output node receives input from
 **  all other nodes in the network.
 **
 **  The third line specifies the unit type. Possible unit types are:
 **  1 for SIGMOIDAL and 2 for ASYMSIGMOIDAL.
 */
/*------------  End of comment   ---------------------------------------*/
/* Network specification begins from the line below the current one. */
XOR with 2 hidden units and no crosscut connections.
1st Connect call connects inputs to both hidden and output, 2nd call connects
hidden unit to the output unit.
#
/* start of net
   Ninputs 2          Nhidden 2         Noutputs 1
   UnitType  1
   Connectcalls 2
   1 2    3 4
   3 4    5 5
   NTrainingPatterns 4
   0.0  0.0 	 -0.5
   0.0  1.0 	 0.5
   1.0  0.0 	 0.5
   1.0  1.0 	 -0.5 
   NTestPatterns 4
   0.0  0.0 	 -0.5
   0.0  1.0 	 0.5
   1.0  0.0 	 0.5
   1.0  1.0 	 -0.5 
   end of second example */
/* 
   This is a sample input network.  All characters before the pound sign
   below are ignored.
#
Ninputs 4          Nhidden 6          Noutputs 1
UnitType 2
Connectcalls 2
1 4     5 10
5 10   11 11
NTrainingPatterns 36
0.33 0.33 0.33 0.33     1.0
0.67 0.33 0.33 0.33     0.0
1.00 0.33 0.33 0.33     0.0
0.33 0.33 0.67 0.33     0.0
0.67 0.33 0.67 0.33     1.0
1.00 0.33 0.67 0.33     0.0
0.33 0.33 1.00 0.33     0.0
0.67 0.33 1.00 0.33     0.0
1.00 0.33 1.00 0.33     1.0

0.33 0.33 0.33 0.67     1.0
0.67 0.33 0.33 0.67     1.0
1.00 0.33 0.33 0.67     0.0
0.33 0.33 0.67 0.67     0.0
0.67 0.33 0.67 0.67     1.0
1.00 0.33 0.67 0.67     1.0

0.33 0.33 0.33 1.00     1.0
0.67 0.33 0.33 1.00     1.0
1.00 0.33 0.33 1.00     1.0

0.33 0.67 0.33 0.33     0.0
0.67 0.67 0.33 0.33     0.0
0.33 0.67 0.67 0.33     0.0
0.67 0.67 0.67 0.33     0.0
0.33 0.67 1.00 0.33     0.0
0.67 0.67 1.00 0.33     0.0

0.33 0.67 0.33 0.67     1.0
0.67 0.67 0.33 0.67     0.0
0.33 0.67 0.67 0.67     0.0
0.67 0.67 0.67 0.67     1.0

0.33 0.67 0.33 1.00     1.0
0.67 0.67 0.33 1.00     1.0

0.33 1.00 0.33 0.33     0.0
0.33 1.00 0.67 0.33     0.0
0.33 1.00 1.00 0.33     0.0

0.33 1.00 0.33 0.67     0.0
0.33 1.00 0.67 0.67     0.0

0.33 1.00 0.33 1.00     1.0
NTestPatterns 0
end of last example */

#endif /*  YOU_WANT_TO_COMPILE_JUNK */

/* end of quickprop1.c */
