"""
python instanton_rate_postproc.py  -c instanton -f <folder_to_store> -t <temperature> -n <half_ring_polymer_number> 
"""
import numpy as np 
import os 
import argparse


# ---- parse input from command line -------
def parse_input():
    # INPUT
    parser = argparse.ArgumentParser(
        description= """
        Postprocessing file to compute either instanton rate / TST rate / tunneling factor. 
                                              instanton / TST/ tunneling_factor 
        """
    )
    
    parser.add_argument(
        "-c",
        "--case",
        default= False,
        help= "Type of calculation to analyse. Options: instanton, TST, tunneling_factor "  
        )
    
    parser.add_argument(
        "-f",
        "--folder",
        default= "",
        help= "the folder that store the calculated rate."
    )

    parser.add_argument(
        "-t",
        "--temperature",
        default= 0,
        help="the temperature of the rate calculation."
    )

    parser.add_argument(
        "-n",
        "--half_ring_polymer_number",
        default= 1,
        help="the half number of the full ring polymer used in the instanton calculation."
    )

    args = parser.parse_args()
    case = args.case 
    folder= args.folder
    nbeads_half_ring_polymer = int(args.half_ring_polymer_number)
    temperature = float(args.temperature)


    if case not in list(["instanton", "TST", "tunneling_factor"]):
        raise ValueError(
            "We can not identify the case. The valid case is 'instanton', 'TST', 'tunneling_factor' "
        )
    return case, folder, nbeads_half_ring_polymer, temperature  
    

def compute_instanton_rate(instanton_folder, nbeads_half_rp, temperature):
    '''
    Compute instanton rate using the data we obtained from post processing
    Data we need to put in:  
    (1) for ring polymer and general info : nbeads_half_rp. temperature
    (2) from reactant file :  Qr_trans, Qr_rot, log_Qvib_rp 
    (3) from instanton file: 1. S/hbar  2. Qinst_trans, 3. Qinst_rot, 4. log(Qvib * N) (log_Qinst_vib_N), 5. BN 
    '''
    # unit conversion. Ktoau  : in Instanton_postprocess.py, temperature is in a.u.. 1 a.u. = 3.15775 * 10^{5}
    hbar = 1 # in atomic unit
    kb = 1  # in atomic unit
    K_to_au = 1 / (3.15775 * np.power(10.0, 5)) 
    rate_au_to_inversecond = 1 / (2.4189 * np.power(10.0, -17))  # rate from atomic unit to seconds

    nbeads_half_rp = nbeads_half_rp  # number of beads for half ring polymer (used in instanton optimization)
    temperature = temperature  # in unit of kevin

    instanton_folder_path = instanton_folder
    instanton_file_name = "rate_N={}_T={}.txt".format(nbeads_half_rp, temperature)
    instanton_file_path = os.path.join(instanton_folder_path, instanton_file_name)

    inp = input("compute rate using instanton theory (result from post processing). instanton folder path {}  yes / no? ".format(instanton_folder_path))
    if inp != "yes":
        print("Abort")
        return 
    else:
        print("Proceed with calculation.")

    nbeads_full_rp = nbeads_half_rp * 2  # number of beads for full ring polymer
    temperature_in_au = temperature * K_to_au  # temperature in atomic unit. 
    beta = 1 / (temperature * K_to_au)  # beta in atomic unit.
    betaN = beta / nbeads_full_rp   # beta / N . atomic unit.

    # ----------- reactant -----------
    # the data below are obtained from reactant post processing using Instantoon_postprocess.py -c reactant
    # partition function for reactant translational / rotational / vibrational dof
    Qr_trans = 359.660
    Qr_rot =  3163811.035
    log_Qr_vib_rp = -312.387 # log(Q_vib) for reactant.


    # ---------- instanton ---------------
    # the data below are obtained from instanton postprocessing using Instanton_postprocess.py -c instanton
    S_over_hbar = 2.829
    Qinst_trans =  359.660
    Qinst_rot = 3146644.86
    # compute log(Q_vib) for instanton 
    log_Qinst_vib_N = -305.639 # log(N^{1+f0} \prod ( 1/(beta_N * hbar * \eta_k) )) here f0 = 6 for polyatomic case
    BN_N = 1636.982 # BN * N (N = nbeads_full_rp)

    Qinst_vib_prefactor = np.sqrt( 2 * np.pi * BN_N / (beta) )  # in postprocess script, hbar = 1. 

    log_Qinst_vib = log_Qinst_vib_N + np.log(Qinst_vib_prefactor)  # log(Q_vib) for instanton 


    # compute ratio of partition function and rate
    Q_trans_ratio = Qinst_trans / Qr_trans
    Q_rot_ratio = Qinst_rot / Qr_rot
    log_Q_vib_ratio = log_Qinst_vib - log_Qr_vib_rp

    rate_in_au = (1 / (2 * np.pi * beta) ) * Q_trans_ratio * Q_rot_ratio *  np.exp( log_Q_vib_ratio - S_over_hbar)  # rate is in atomic unit.

    rate = rate_in_au * rate_au_to_inversecond  # rate in s^-1. 

    rate = rate / np.power(10.0, 12)  # rate in ps^-1

    # write result to the output file
    with open(instanton_file_path , "w") as f:
        f.write( 
            "{:8s}  |   {:8s}   |   {:8s}   |   {:8s}  |  {:8s}  \n".format(
                "Qinst_trans / Qr_trans", 
                "Qinst_rot / Qr_rot",
                "log( Qinst_vib / Qr_vib )",
                "S/hbar",
                "rate (ps^-1)"        
            ) 
        )

        f.write(
            " {:8.3f}  |   {:8.3f}   |  {:8.3f}  |  {:8.3f}  |  {:8.3f}  \n".format(
                Q_trans_ratio,  Q_rot_ratio,  log_Q_vib_ratio,  S_over_hbar, rate
            )
        )
    
    return rate

def compute_rate_Erying_formula(TS_folder, temperature):
    '''
    compute the rate of calculation using Erying's TS theory formula. See eq(3) in 10.1021/acs.jpclett.6b02115. Note the denominator here should be beta.
    '''
    hbar = 1 # in atomic unit
    kb = 1  # in atomic unit
    K_to_au = 1 / (3.15775 * np.power(10.0, 5)) 
    rate_au_to_inversecond = 1 / (2.4189 * np.power(10.0, -17))  # rate from atomic unit to seconds

    TS_folder_path = TS_folder
    rate_file_name = "rate_Erying_TST.txt"
    rate_file_path = os.path.join(TS_folder_path, rate_file_name)

    inp = input("compute rate using Erying's TS theory. TS folder path {}  yes / no? ".format(TS_folder_path))
    if inp != "yes":
        print("Abort")
        return 
    else:
        print("Proceed with calculation.")

    temperature = temperature  # temperature in K unit.

    beta = 1 / (temperature * K_to_au) # inverse of temperature in a.u. unit 

    # ----------- reactant -----------
    # the data below are obtained from reactant post processing using Instantoon_postprocess.py -c reactant
    # partition function for reactant translational / rotational / vibrational dof
    Qr_trans = 33.774 
    Qr_rot = 27140.466
    log_Qvib_rp = -158.515  # log(Q_vib)

    #---------- Transition state ---------
    # data below are obtained from TS postprocessing using Instanton_postprocess.py -c TS
    # partition function for transition state (TS) translational / rotational / vibrational dof
    Q_TS_trans = 67.341
    Q_TS_rot = 47424.15
    log_Q_TS_vib = -81.1135 
    betaV = 7.17498 

    # compute ratio of partition function and rate
    Q_trans_ratio = Q_TS_trans / Qr_trans
    Q_rot_ratio = Q_TS_rot / Qr_rot
    log_Q_vib_ratio = log_Q_TS_vib - log_Qvib_rp 
    
    rate = 1/(2 * np.pi * beta) * Q_trans_ratio * Q_rot_ratio * np.exp( log_Q_vib_ratio - betaV) # rate in atomic unit

    rate = rate * rate_au_to_inversecond    # rate in unit of s^-1
    rate = rate * np.power(10.0, -12)  # rate in unit of ps^-1

    with open(rate_file_path , "w") as f:
        f.write( 
            "{:8s}  |   {:8s}   |   {:8s}   |   {:8s}  |  {:8s}  \n".format(
                "Qinst_trans / Qr_trans", 
                "Qinst_rot / Qr_rot",
                "log( Qinst_vib / Qr_vib )",
                "betaV",
                "rate (ps^-1)"        
            ) 
        )

        f.write(
            " {:8.3f}  |   {:8.3f}   |  {:8.3f}  |  {:8.3f}  |  {:8.6f}  \n".format(
                Q_trans_ratio,  Q_rot_ratio,  log_Q_vib_ratio,  betaV, rate
            )
        )


def compute_tunneling_factor(instanton_folder, nbeads_half_rp, temperature):
    '''
    compute tunneling factor, k_{instanton} / k_{TST}: rate of instanton / rate of Erying's transition state theory.
    '''
    hbar = 1 # in atomic unit
    kb = 1  # in atomic unit
    K_to_au = 1 / (3.15775 * np.power(10.0, 5)) 
    rate_au_to_inversecond = 1 / (2.4189 * np.power(10.0, -17))  # rate from atomic unit to seconds


    nbeads_half_rp = nbeads_half_rp  # number of beads for half ring polymer (used in instanton optimization)
    temperature = temperature  # in unit of kevin

    instanton_folder_path = instanton_folder
    tunneling_factor_file_name = "tunneling_factor_N={}_T={}.txt".format(nbeads_half_rp, temperature)
    tunneling_factor_file_path = os.path.join(instanton_folder_path, tunneling_factor_file_name)

    inp = input("compute tunneling factor using instanton theory (result from post processing). tunneling factor folder path {}  yes / no? ".format(instanton_folder_path))
    if inp != "yes":
        print("Abort")
        return 
    else:
        print("Proceed with calculation.")

    nbeads_full_rp = nbeads_half_rp * 2  # number of beads for full ring polymer
    temperature_in_au = temperature * K_to_au  # temperature in atomic unit. 
    beta = 1 / (temperature * K_to_au)  # beta in atomic unit.
    betaN = beta / nbeads_full_rp   # beta / N . atomic unit.

    # ---------- instanton ---------------
    # the data below are obtained from instanton postprocessing using Instanton_postprocess.py -c instanton
    S_over_hbar = 41.514
    Qinst_trans = 3.602
    Qinst_rot = 556.953
    
    # compute log(Q_vib) for instanton 
    log_Qinst_vib_term = -88.912 # log(N^{1+f0} \prod ( 1/(beta_N * hbar * \eta_k) )) here f0 = 6 for polyatomic case

    BN_N = 35306.390  # BN * N (N = nbeads_full_rp)

    Qinst_vib_prefactor = np.sqrt( 2 * np.pi * BN_N / (beta) )  # in postprocess script, hbar = 1. 

    log_Qinst_vib = log_Qinst_vib_term + np.log(Qinst_vib_prefactor)  # log(Q_vib) for instanton 

    #---------- Transition state ---------
    # data below are obtained from TS postprocessing using Instanton_postprocess.py -c TS
    # partition function for transition state (TS) translational / rotational / vibrational dof
    Q_TS_trans = 3.602
    Q_TS_rot = 426.439
    log_Q_TS_vib = -88.887
    betaV = 50.331

    #---- Tunneling factor -------------
    Q_trans_ratio = Qinst_trans / Q_TS_trans 
    Q_rot_ratio = Qinst_rot / Q_TS_rot 
    log_Q_vib_ratio = log_Qinst_vib - log_Q_TS_vib

    tunneling_factor = Q_trans_ratio * Q_rot_ratio * np.exp(log_Q_vib_ratio - (S_over_hbar - betaV))

    #printing out the transmission factor and the relevant contributions. 
    with open(tunneling_factor_file_path, "w") as f:
        f.write('f_trans = %8.5e'%Q_trans_ratio + "\n")
        f.write('f_rot = %8.5e'%Q_rot_ratio + "\n")
        f.write('f_vib = %8.5e'%np.exp(log_Q_vib_ratio) + "\n")
        f.write('exp(-S/hbar) = %8.5e'%np.exp(-S_over_hbar) + "\n")
        f.write('exp(V/beta) = %8.5e'%np.exp(betaV) + "\n")
        f.write('============================= \n')
        f.write('kappa = %8.5e'%tunneling_factor + "\n")

    return tunneling_factor

def compute_result():
    calculation_case, folder_path, nbeads_half_rp, temperature = parse_input()

    if calculation_case == 'instanton':
        compute_instanton_rate(folder_path, nbeads_half_rp, temperature)
    elif calculation_case == 'TST':
        compute_rate_Erying_formula(folder_path, temperature)
    elif calculation_case == 'tunneling_factor':
        compute_tunneling_factor(folder_path, nbeads_half_rp, temperature)
    else:
        raise ValueError("Can not recognize the case")

compute_result()