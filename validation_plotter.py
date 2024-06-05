import numpy as np
# print(np.__file__)
import awkward as ak
import dask_awkward as dak
import json
import argparse
import os
from ROOT_utils import setTDRStyle, CMS_lumi, reweightROOTH
import dask.dataframe as dd
from distributed import Client
import time
import tqdm

# real process arrangement
group_data_processes = ["data_A", "data_B", "data_C", "data_D",]
# group_DY_processes = ["dy_M-100To200", "dy_M-50"] # dy_M-50 is not used in ggH BDT training input
group_DY_processes = ["dy_M-100To200"]
group_Top_processes = ["ttjets_dl", "ttjets_sl", "st_tw_top", "st_tw_antitop"]
group_Ewk_processes = ["ewk_lljj_mll50_mjj120"]
group_VV_processes = ["ww_2l2nu", "wz_3lnu", "wz_2l2q", "wz_1l1nu2q", "zz"]# diboson
group_ggH_processes = ["ggh_powheg"]
group_VBF_processes = ["vbf_powheg"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="string value year",
    )
    parser.add_argument(
    "-data",
    "--data",
    dest="data_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of data samples represented by alphabetical letters A-H",
    )
    parser.add_argument(
    "-bkg",
    "--background",
    dest="bkg_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of bkg samples represented by shorthands: DY, TT, ST, DB (diboson), EWK",
    )
    parser.add_argument(
    "-sig",
    "--signal",
    dest="sig_samples",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of sig samples represented by shorthands: ggH, VBF",
    )
    parser.add_argument(
    "-var",
    "--variables",
    dest="variables",
    default=[],
    nargs="*",
    type=str,
    action="store",
    help="list of variables to plot (ie: jet, mu, dimuon)",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/hmm/test_validation/stage1_output/",
    action="store",
    help="load path",
    )
    parser.add_argument(
    "-save",
    "--save_path",
    dest="save_path",
    default="./figs/",
    action="store",
    help="save path",
    )
    parser.add_argument(
    "-lumi",
    "--lumi",
    dest="lumi",
    default="",
    action="store",
    help="string value of integrated luminosity to label",
    )
    parser.add_argument(
    "--status",
    dest="status",
    default="",
    action="store",
    help="Status of results ie Private, Preliminary, In Progress",
    )
    parser.add_argument(
    "--no_ratio",
    dest="no_ratio",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="doesn't plot Data/MC ratio",
    )
    parser.add_argument(
    "--ROOT_style",
    dest="ROOT_style",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, uses pyROOT functionality instead of mplhep",
    )
    parser.add_argument(
    "--linear_scale",
    dest="linear_scale",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, provide plots in linear scale",
    )
    parser.add_argument(
    "-reg",
    "--region",
    dest="region",
    default="signal",
    action="store",
    help="region value to plot, available regions are: h_peak, h_sidebands, z_peak and signal (h_peak OR h_sidebands)",
    )
    parser.add_argument(
    "--vbf",
    dest="vbf_cat_mode",
    default=False, 
    action=argparse.BooleanOptionalAction,
    help="If true, apply vbf cut for vbf category, else, ggH category cut",
    )
    #---------------------------------------------------------
    # gather arguments
    args = parser.parse_args()
    year = args.year
    available_processes = []
    # take data
    data_samples = args.data_samples
    if len(data_samples) >0:
        for data_letter in data_samples:
            available_processes.append(f"data_{data_letter.upper()}")
    # take bkg
    bkg_samples = args.bkg_samples
    if len(bkg_samples) >0:
        for bkg_sample in bkg_samples:
            if bkg_sample.upper() == "DY": # enforce upper case to prevent confusion
                # available_processes.append("dy_M-50")
                available_processes.append("dy_M-100To200")
            elif bkg_sample.upper() == "TT": # enforce upper case to prevent confusion
                available_processes.append("ttjets_dl")
                available_processes.append("ttjets_sl")
            elif bkg_sample.upper() == "ST": # enforce upper case to prevent confusion
                available_processes.append("st_tw_top")
                available_processes.append("st_tw_antitop")
            elif bkg_sample.upper() == "VV": # enforce upper case to prevent confusion
                available_processes.append("ww_2l2nu")
                available_processes.append("wz_3lnu")
                available_processes.append("wz_2l2q")
                available_processes.append("wz_1l1nu2q")
                available_processes.append("zz")
            elif bkg_sample.upper() == "EWK": # enforce upper case to prevent confusion
                available_processes.append("ewk_lljj_mll50_mjj120")
            else:
                print(f"unknown background {bkg_sample} was given!")
        
    # take sig
    sig_samples = args.sig_samples
    if len(sig_samples) >0:
        for sig_sample in sig_samples:
            if sig_sample.upper() == "GGH": # enforce upper case to prevent confusion
                available_processes.append("ggh_powheg")
            elif sig_sample.upper() == "VBF": # enforce upper case to prevent confusion
                available_processes.append("vbf_powheg")
            else:
                print(f"unknown signal {sig_sample} was given!")
    # gather variables to plot:
    kinematic_vars = ['pt', 'eta', 'phi']
    variables2plot = []
    if len(args.variables) == 0:
        print("no variables to plot!")
        raise ValueError
    for particle in args.variables:
        if "dimuon" in particle:
            variables2plot.append(f"{particle}_mass")
            variables2plot.append(f"{particle}_pt")
            variables2plot.append(f"{particle}_cos_theta_cs")
            variables2plot.append(f"{particle}_phi_cs")
            variables2plot.append(f"mmj_min_dPhi_nominal")
            variables2plot.append(f"mmj_min_dEta_nominal")
            
        elif "dijet" in particle:
            # variables2plot.append(f"gjj_mass_nominal")
            variables2plot.append(f"jj_mass_nominal")
            variables2plot.append(f"jj_dEta_nominal")
            variables2plot.append(f"jj_dPhi_nominal")
        elif ("mu" in particle) :
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}")
                variables2plot.append(f"{particle}2_{kinematic}")
        elif ("jet" in particle):
            variables2plot.append(f"njets_nominal")
            for kinematic in kinematic_vars:
                # plot both leading and subleading muons/jets
                variables2plot.append(f"{particle}1_{kinematic}_nominal")
                variables2plot.append(f"{particle}2_{kinematic}_nominal")
        else:
            print(f"Unsupported variable: {particle} is given!")
    print(f"variables2plot: {variables2plot}")
    # obtain plot settings from config file
    with open("./plot_settings.json", "r") as file:
        plot_settings = json.load(file)
    status = args.status.replace("_", " ")
    
    # define client for parallelization for speed boost
    client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='4 GiB') 
    print("Local scale Client created")
    # record time
    time_step = time.time()

    # print(f"available_processes: {available_processes}")
    # loaded_events = {} # intialize dictionary containing all the arrays
    # for process in tqdm.tqdm(available_processes):
    #     print(f"process: {process}")
    #     full_load_path = args.load_path+f"/{year}/{process}/*.parquet"
    #     # events = dak.from_parquet(full_load_path) 
    #     events = dd.read_parquet(full_load_path)
    #     fields2load = variables2plot + ["wgt_nominal", "region", "nBtagLoose_nominal", "nBtagMedium_nominal",]
    #     is_data = "data" in process.lower()
    #     if not is_data: # MC sample
    #          fields2load += ["gjj_mass_nominal", "gjj_dR_nominal",]
    #     events = events[fields2load]
    #     # load data to memory using compute()
    #     events = ak.zip({
    #         field : events[field] for field in events.columns
    #     }).compute()
    #     loaded_events[process] = events
    #     print(f"events: {events}")
        
    if args.ROOT_style:
        import ROOT
        #Plotting part
        setTDRStyle()
        canvas = ROOT.TCanvas("canvas","",600,750);
        canvas.cd();
        
        pad = ROOT.TPad("pad","pad",0,0.,1,1);
        pad.SetFillColor(0);
        pad.SetFillStyle(0);
        pad.SetTickx(1);
        pad.SetTicky(1);
        pad.SetBottomMargin(0.3);
        pad.SetRightMargin(0.06);
        pad.Draw();
        pad.cd();
        counter = 0
        for var in tqdm.tqdm(variables2plot):
            print(f"var: {var}")
            counter +=1
            if counter % 5 ==0:
                print("restarting client!")
                client.restart(wait_for_workers=False)
            var_step = time.time()
            if var not in plot_settings.keys():
                print(f"variable {var} not configured in plot settings!")
                continue
            binning = np.linspace(*plot_settings[var]["binning_linspace"])
            group_data_hists = []
            group_DY_hists = []
            group_Top_hists = []
            group_Ewk_hists = []
            group_VV_hists = []
            group_other_hists = []
            group_ggH_hists = [] # there should only be one ggH histogram, but making a list for consistency
            group_VBF_hists = [] # there should only be one VBF histogram, but making a list for consistency
            

            print(f"available_processes: {available_processes}")
            for process in available_processes:
                print(f"process: {process}")
                full_load_path = args.load_path+f"/{year}/{process}/*.parquet"
                # events = dak.from_parquet(full_load_path) 
                events = dd.read_parquet(full_load_path)
                # fraction_weight = 1/events.fraction[0].compute()
                # print(f"process: {process}")
                # events = loaded_events[process]
                fraction_weight = 1.0

                # test start -----------------------------------------------------------
                # collect weights
                # weights = events["wgt_nominal"].compute().to_numpy()
                weights = events["wgt_nominal"].to_numpy()
                vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)
                # forgot add region so recalculat from dimuon mass
                mass = events.dimuon_mass
                vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)
                vbf_cut.fillna(False)
                if args.region == "signal":
                    region = (events.region=="h-sidebands") | (events.region=="h-peak")
                elif args.region == "h_peak":
                    region = (events.region=="h-peak") 
                elif args.region == "h_sidebands":
                    region = (events.region=="h-sidebands") 
                elif args.region == "z_peak":
                    region = (events.region=="z-peak") 
                elif args.region == "all":
                    region = (events.region=="h-sidebands") | (events.region=="h-peak") | (events.region=="z-peak")
                else: 
                    print("ERROR: not acceptable region!")
                    raise ValueError

                if args.vbf_cat_mode:
                    print("vbf mode!")
                    prod_cat_cut =  vbf_cut
                    # apply additional cut to MC samples if vbf 
                    # VBF filter cut start -------------------------------------------------
                    # if process == "dy_VBF_filter":
                    #     print("dy_VBF_filter extra!")
                    #     prod_cat_cut =  ( prod_cat_cut  
                    #                 & ak.fill_none((events.gjj_mass > 350), value=False) 
                    #     )
                    # elif process == "dy_M-100To200":
                    #     print("dy_M-100To200 extra!")
                    #     prod_cat_cut =  ( prod_cat_cut  
                    #                 & ak.fill_none((events.gjj_mass <= 350), value=False)  
                    #     )
                    # VBF filter cut end -------------------------------------------------
                else: # we're interested in ggH category
                    print("ggH mode!")
                    prod_cat_cut =  ~vbf_cut
                    
                btag_cut =(events.nBtagLoose_nominal >= 2) | (events.nBtagMedium_nominal >= 1)
                category_selection = (
                    prod_cat_cut & 
                    region &
                    ~btag_cut # btag cut is for VH and ttH categories
                )
                # ).compute()
                # test end --------------------------------------------------------


                
                category_selection = category_selection.to_numpy()
                weights = weights*category_selection
                # values = events[var].fillna(-999.0).compute().to_numpy()
                values = events[var].fillna(-999.0).to_numpy()
                values_filter = values!=-999.0
                values = values[values_filter]
                weights = weights[values_filter]
                np_hist, _ = np.histogram(values, bins=binning, weights = weights)
                # collect same histogram, but for weight squares for error calculation 
                np_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
                
                hist_errs = np.sqrt(np_hist_w2)
                if process in group_data_processes:
                    print("data activated")
                    # var_hist_data = ROOT.TH1F( var+'_hist_data', var, len(binning)-1, min(binning), max(binning))
                    var_hist_data = ROOT.TH1F(process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_data.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_data.SetBinContent(1+idx, np_hist[idx])
                        var_hist_data.SetBinError(1+idx, hist_errs[idx])
                    group_data_hists.append(var_hist_data)
                #-------------------------------------------------------
                elif process in group_DY_processes:
                    print("DY activated")
                    # var_hist_DY = ROOT.TH1F( var+'_hist_DY', var, len(binning)-1, min(binning), max(binning))
                    var_hist_DY = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_DY.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_DY.SetBinContent(1+idx, np_hist[idx])
                        var_hist_DY.SetBinError(1+idx, hist_errs[idx])
                    group_DY_hists.append(var_hist_DY)
                #-------------------------------------------------------
                elif process in group_Top_processes:
                    print("top activated")
                    # var_hist_Top = ROOT.TH1F( var+'_hist_Top', var, len(binning)-1, min(binning), max(binning))
                    var_hist_Top = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_Top.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_Top.SetBinContent(1+idx, np_hist[idx])
                        var_hist_Top.SetBinError(1+idx, hist_errs[idx])
                    group_Top_hists.append(var_hist_Top)
                #-------------------------------------------------------
                elif process in group_Ewk_processes:
                    print("Ewk activated")
                    # var_hist_Ewk = ROOT.TH1F( var+'_hist_Ewk', var, len(binning)-1, min(binning), max(binning))
                    var_hist_Ewk = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_Ewk.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_Ewk.SetBinContent(1+idx, np_hist[idx])
                        var_hist_Ewk.SetBinError(1+idx, hist_errs[idx])
                    group_Ewk_hists.append(var_hist_Ewk)
                #-------------------------------------------------------
                elif process in group_VV_processes:
                    print("VV activated")
                    # var_hist_VV = ROOT.TH1F( var+'_hist_VV', var, len(binning)-1, min(binning), max(binning))
                    var_hist_VV = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_VV.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_VV.SetBinContent(1+idx, np_hist[idx])
                        var_hist_VV.SetBinError(1+idx, hist_errs[idx])
                    group_VV_hists.append(var_hist_VV)
                #-------------------------------------------------------
                elif process in group_ggH_processes:
                    print("ggH activated")
                    # var_hist_ggH = ROOT.TH1F( var+'_hist_ggH', var, len(binning)-1, min(binning), max(binning))
                    var_hist_ggH = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_ggH.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_ggH.SetBinContent(1+idx, np_hist[idx])
                        var_hist_ggH.SetBinError(1+idx, hist_errs[idx])
                    group_ggH_hists.append(var_hist_ggH)
                #-------------------------------------------------------
                elif process in group_VBF_processes:
                    print("VBF activated")
                    # var_hist_VBF = ROOT.TH1F( var+'_hist_VBF', var, len(binning)-1, min(binning), max(binning))
                    var_hist_VBF = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_VBF.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_VBF.SetBinContent(1+idx, np_hist[idx])
                        var_hist_VBF.SetBinError(1+idx, hist_errs[idx])
                    group_VBF_hists.append(var_hist_VBF)
                #-------------------------------------------------------
                else: # put into "other" bkg group
                    # if "dy_M-50" in process:
                    #     # print("dy_M-50 activated")
                    #     continue
                    print("other activated")
                    # var_hist_other = ROOT.TH1F( var+'_hist_other', var, len(binning)-1, min(binning), max(binning))
                    var_hist_other = ROOT.TH1F( process, var, len(binning)-1, min(binning), max(binning))
                    var_hist_other.Sumw2()
                    for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                        var_hist_other.SetBinContent(1+idx, np_hist[idx])
                        var_hist_other.SetBinError(1+idx, hist_errs[idx])
                    group_other_hists.append(var_hist_other)
        
            dummy_hist = ROOT.TH1F('dummy_hist', "dummy", len(binning)-1, min(binning), max(binning))
            dummy_hist.Sumw2() # not sure if this is necessary, but just in cas
            dummy_hist.GetXaxis().SetTitleSize(0);
            dummy_hist.GetXaxis().SetLabelSize(0);
            dummy_hist.GetYaxis().SetTitle("Events")
            dummy_hist.Draw("EP");
            
            all_MC_hist_list = []
            
            if len(group_DY_hists) > 0:
                DY_hist_stacked = group_DY_hists[0]
                if len(group_DY_hists) > 1:
                    for idx in range(1, len(group_DY_hists)):
                        DY_hist_stacked.Add(group_DY_hists[idx])
                DY_hist_stacked.SetLineColor(1);
                DY_hist_stacked.SetFillColor(ROOT.kOrange+1);
                all_MC_hist_list.append(DY_hist_stacked)
            #----------------------------------------------
            if len(group_Top_hists) > 0:
                Top_hist_stacked = group_Top_hists[0]
                if len(group_Top_hists) > 1:
                    for idx in range(1, len(group_Top_hists)):
                        Top_hist_stacked.Add(group_Top_hists[idx])
                Top_hist_stacked.SetLineColor(1);
                Top_hist_stacked.SetFillColor(ROOT.kGreen+1);
                all_MC_hist_list.append(Top_hist_stacked)
            #----------------------------------------------
            if len(group_Ewk_hists) > 0:
                Ewk_hist_stacked = group_Ewk_hists[0]
                if len(group_Ewk_hists) > 1:
                    for idx in range(1, len(group_Ewk_hists)):
                        Ewk_hist_stacked.Add(group_Ewk_hists[idx])
                Ewk_hist_stacked.SetLineColor(1);
                Ewk_hist_stacked.SetFillColor(ROOT.kMagenta+1);
                all_MC_hist_list.append(Ewk_hist_stacked)
            #----------------------------------------------
            if len(group_VV_hists) > 0:
                VV_hist_stacked = group_VV_hists[0]
                if len(group_VV_hists) > 1:
                    for idx in range(1, len(group_VV_hists)):
                        VV_hist_stacked.Add(group_VV_hists[idx])
                VV_hist_stacked.SetLineColor(1);
                VV_hist_stacked.SetFillColor(ROOT.kAzure+1);
                all_MC_hist_list.append(VV_hist_stacked)
            #----------------------------------------------
            if len(group_other_hists) > 0:
                other_hist_stacked = group_other_hists[0]
                if len(group_other_hists) > 1:
                    for idx in range(1, len(group_other_hists)):
                        other_hist_stacked.Add(group_other_hists[idx])
                other_hist_stacked.SetLineColor(1);
                other_hist_stacked.SetFillColor(ROOT.kGray);
                all_MC_hist_list.append(other_hist_stacked)
            #----------------------------------------------
            
            
            # separately make copy of mc hists for ratio calculation. doing it directly onto THStack is a pain
            all_MC_hist_copy = all_MC_hist_list[0].Clone("all_MC_hist_copy");# we assume that there's at least one element in all_MC_hist_list
            all_MC_hist_copy.Sumw2() 
            for idx in range(1, len(all_MC_hist_list)):
                all_MC_hist_copy.Add(all_MC_hist_list[idx]) 
            
            # aggregate all MC hist by stacking them and then plot
            all_MC_hist_stacked = ROOT.THStack("all_MC_hist_stacked", "");
            if len(all_MC_hist_list) > 0:
                all_MC_hist_list.reverse() # add smallest histgrams first, so from other -> DY
                for MC_hist_stacked in all_MC_hist_list: 
                    all_MC_hist_stacked.Add(MC_hist_stacked) 
                for idx in range(all_MC_hist_stacked.GetStack().GetEntries()):
                    all_MC_hist = all_MC_hist_stacked.GetStack().At(idx) # get the TH1F portion of THStack
                all_MC_hist_stacked.Draw("hist same");
            
            # stack and plot data 
            if len(group_data_hists) > 0:
                data_hist_stacked = group_data_hists[0]
                data_hist_stacked.Sumw2()
                print(f"data_hist_stacked: {data_hist_stacked}")
                if len(group_data_hists) > 1:
                    for idx in range(1, len(group_data_hists)):
                        data_hist_stacked.Add(group_data_hists[idx])
                        print(f"group_data_hists[idx]: {group_data_hists[idx]}")
            
                # decorate the data_histogram
                xlabel = plot_settings[var]["xlabel"].replace('$', '')
                data_hist_stacked.GetXaxis().SetTitle(xlabel);
                data_hist_stacked.GetXaxis().SetTitleOffset(1.10);
                data_hist_stacked.GetYaxis().SetTitleOffset(1.15);
            
                data_hist_stacked.SetMarkerStyle(20);
                data_hist_stacked.SetMarkerSize(1);
                data_hist_stacked.SetMarkerColor(1);
                data_hist_stacked.SetLineColor(1);
                data_hist_stacked.Draw("EPsame");        
            
            
            # plot signals: ggH and VBF
            if len(group_ggH_hists) > 0:
                hist_ggH = group_ggH_hists[0]
                hist_ggH.Sumw2()
                hist_ggH.SetLineColor(ROOT.kBlack);
                hist_ggH.SetLineWidth(3);
                hist_ggH.Sumw2()
                hist_ggH.Draw("hist same");
            if len(group_VBF_hists) > 0:
                hist_VBF = group_VBF_hists[0]
                hist_VBF.Sumw2()
                hist_VBF.SetLineColor(ROOT.kRed);
                hist_VBF.SetLineWidth(3);
                hist_VBF.Draw("hist same");
        
            # Ratio pad
            if not args.no_ratio:
                pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9);
                pad2.SetFillColor(0);
                pad2.SetGridy(1);
                pad2.SetFillStyle(0);
                pad2.SetTickx(1);
                pad2.SetTicky(1);
                pad2.SetTopMargin(0.7);
                pad2.SetRightMargin(0.06);
                pad2.Draw();
                pad2.cd();
                
                if (len(group_data_hists) > 0) and (len(all_MC_hist_list) > 0):
                    print("ratio activated")
                    num_hist = data_hist_stacked.Clone("num_hist");
                    den_hist = all_MC_hist_stacked.Clone("den_hist").GetStack().Last(); # to get TH1F from THStack, one needs to call .GetStack().Last()
                    # print(num_hist)
                    # print(den_hist)
                    num_hist.Divide(den_hist); # we assume Sumw2 mode was previously activated
                    num_hist.SetStats(ROOT.kFALSE);
                    num_hist.SetLineColor(ROOT.kBlack);
                    num_hist.SetMarkerColor(ROOT.kBlack);
                    num_hist.SetMarkerSize(0.8);
                    
                    # get MC statistical errors 
                    mc_ratio = all_MC_hist_copy.Clone("mc_ratio")                    # set all of its errors to zero to prevent double counting of same error
                    for idx in range(1, mc_ratio.GetNbinsX()+1):
                        mc_ratio.SetBinError(idx, 0)
                    mc_ratio.Divide(den_hist) # divide by itself, errors from den_hist are propagated
                    mc_ratio.SetLineColor(0);
                    mc_ratio.SetMarkerColor(0);
                    mc_ratio.SetMarkerSize(0);
                    mc_ratio.SetFillColor(ROOT.kGray);
                
                    # get ratio line 
                    ratio_line = data_hist_stacked.Clone("num_hist");
                    for idx in range(1, mc_ratio.GetNbinsX()+1):
                        ratio_line.SetBinContent(idx, 1)
                        ratio_line.SetBinError(idx, 0)
                    ratio_line.SetMarkerSize(0);
                    ratio_line.SetLineColor(ROOT.kBlack);
                    ratio_line.SetLineStyle(2);
                    ratio_line.SetFillColor(0);
                    ratio_line.GetYaxis().SetTitle("Data/Pred.");
                    ratio_line.GetYaxis().SetRangeUser(0.5,1.5);
                    ratio_line.GetYaxis().SetTitleSize(num_hist.GetYaxis().GetTitleSize()*0.85);
                    ratio_line.GetYaxis().SetLabelSize(num_hist.GetYaxis().GetLabelSize()*0.85);
                    ratio_line.GetYaxis().SetNdivisions(505);
                
                    ratio_line.Draw("SAME");
                    mc_ratio.Draw("E2 SAME");
                    num_hist.Draw("PE1 SAME");
                    pad2.RedrawAxis("sameaxis");
        
            # setup legends
            if args.no_ratio:
                leg = ROOT.TLegend(0.40,0.70,0.96,0.9)
            else: # plot ratio
                leg = ROOT.TLegend(0.40,0.80,0.96,1.0)
            
            leg.SetFillColor(0);
            leg.SetFillStyle(0);
            leg.SetBorderSize(0);
            leg.SetNColumns(2);
            if len(group_data_hists) > 0:
                leg.AddEntry(data_hist_stacked,"Data","PEL")
            if len(group_DY_hists) > 0:
                leg.AddEntry(DY_hist_stacked,"DY","F")
            if len(group_Top_hists) > 0:
                leg.AddEntry(Top_hist_stacked,"TOP","F")
            if len(group_Ewk_hists) > 0:
                leg.AddEntry(Ewk_hist_stacked,"Ewk","F")
            if len(group_VV_hists) > 0:
                leg.AddEntry(VV_hist_stacked,"VV","F")
            if len(group_other_hists) > 0:
                leg.AddEntry(other_hist_stacked,"Other","F")
            if len(group_ggH_hists) > 0:
                leg.AddEntry(hist_ggH,"ggH","L")
            if len(group_VBF_hists) > 0:
                leg.AddEntry(hist_VBF,"VBF","L")
            leg.Draw("same");
            
            
            pad.RedrawAxis("sameaxis");
                
            pad.cd();
            if not args.linear_scale:
                dummy_hist.GetYaxis().SetRangeUser(0.01, 1e9);
                pad.SetLogy();
            else:
                binmax = data_hist_stacked.GetMaximumBin()
                max_y = data_hist_stacked.GetBinContent(binmax)
                dummy_hist.GetYaxis().SetRangeUser(0.0, 1.3*max_y);
            pad.Modified();
            pad.Update();
            CMS_lumi(canvas, args.lumi, up=True, reduceSize=True, status=status);
            pad.RedrawAxis("sameaxis");
            full_save_path = f"{args.save_path }/{year}/ROOT/Reg_{args.region}"
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            canvas.SaveAs(f"{full_save_path}/{var}.pdf");

            # record time it took
            var_elapsed = round(time.time() - var_step, 3)
            print(f"Finished processing {var} in {var_elapsed} s.")
    else:
        import mplhep as hep
        import matplotlib.pyplot as plt
        import matplotlib
        hep.style.use("CMS")
        # Dictionary for histograms and binnings


        for var in tqdm.tqdm(variables2plot):
            var_step = time.time()
        # for process in available_processes:
            if var not in plot_settings.keys():
                print(f"variable {var} not configured in plot settings!")
                continue
            if "jet" in var:
                var = var+ "_nominal"
            #-----------------------------------------------
            # intialize variables for filling histograms
            binning = np.linspace(*plot_settings[var]["binning_linspace"])
            group_data_hists = []
            group_DY_hists = []
            group_Top_hists = []
            group_Ewk_hists = []
            group_VV_hists = []
            group_other_hists = []  # histograms not belonging to any other mc bkg group
            group_ggH_hists = [] # there should only be one ggH histogram, but making a list for consistency
            group_VBF_hists = [] # there should only be one VBF histogram, but making a list for consistency

            # collect weight squarted histograms for error calculation
            group_data_hists_w2 = []
            group_DY_hists_w2 = []
            group_Top_hists_w2 = []
            group_Ewk_hists_w2 = []
            group_VV_hists_w2 = []
            group_other_hists_w2 = []
            
            
            print(f"available_processes: {available_processes}")
            for process in available_processes:    
                print(f"process: {process}")
                full_load_path = args.load_path+f"/{year}/{process}/*.parquet"      
                # events = dak.from_parquet(full_load_path)
                events = dd.read_parquet(full_load_path)
                # test start -----------------------------------------------------------
                # collect weights
                weights = events["wgt_nominal"].compute().to_numpy()
                # print(f"events.columns: {events.columns}")
                vbf_cut = (events.jj_mass_nominal > 400) & (events.jj_dEta_nominal > 2.5) & (events.jet1_pt_nominal > 35)
                vbf_cut.fillna(False)
                # forgot add region so recalculat from dimuon mass
                # mass = events.dimuon_mass
                # events["region"] = None
                # print(f"events: {events}")
                # z_peak = ((mass > 76) & (mass < 106))
                # h_sidebands = ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150))
                # h_peak = ((mass > 115.03) & (mass < 135.03))
                # if args.region == "signal":
                #     region = h_sidebands | h_peak
                # elif args.region == "h_peak":
                #     region = h_peak 
                # elif args.region == "h_sidebands":
                #     region = h_sidebands 
                # elif args.region == "z_peak":
                #     region = z_peak 
                # else: 
                #     print("ERROR: not acceptable region!")
                #     raise ValueError
                if args.region == "signal":
                    region = (events.region=="h-sidebands") | (events.region=="h-peak")
                elif args.region == "h_peak":
                    region = (events.region=="h-peak") 
                elif args.region == "h_sidebands":
                    region = (events.region=="h-sidebands") 
                elif args.region == "z_peak":
                    region = (events.region=="z-peak") 
                elif args.region == "all":
                    region = (events.region=="h-sidebands") | (events.region=="h-peak") | (events.region=="z-peak")
                else: 
                    print("ERROR: not acceptable region!")
                    raise ValueError
                    
                if args.vbf_cat_mode:
                    print("vbf mode!")
                    prod_cat_cut =  vbf_cut
                    # apply additional cut to MC samples if vbf 
                    # VBF filter cut start -------------------------------------------------
                    # if process == "dy_VBF_filter":
                    #     print("dy_VBF_filter extra!")
                    #     prod_cat_cut =  ( prod_cat_cut  
                    #                 & ak.fill_none((events.gjj_mass > 350), value=False) 
                    #     )
                    # elif process == "dy_M-100To200":
                    #     print("dy_M-100To200 extra!")
                    #     prod_cat_cut =  ( prod_cat_cut  
                    #                 & ak.fill_none((events.gjj_mass <= 350), value=False)  
                    #     )
                    # VBF filter cut end -------------------------------------------------
                else: # we're interested in ggH category
                    print("ggH mode!")
                    prod_cat_cut =  ~vbf_cut
                    
                btag_cut =(events.nBtagLoose_nominal >= 2) | (events.nBtagMedium_nominal >= 1)
                category_selection = (
                    prod_cat_cut & 
                    region &
                    ~btag_cut # btag cut is for VH and ttH categories
                ).compute()
                # test end ------------------------------------
                
                category_selection = category_selection.to_numpy() # this will be multiplied with weights
                print(f"category_selection.shape: {category_selection.shape}")
                print(f"weights.shape: {weights.shape}")
                weights = weights*category_selection # weights where category_selection==False -> zero
                # print(f"weights.shape: {weights[weights>0].shape}")
                # print(f"(events[var]: {events[var].compute()}")
                values = events[var].fillna(-999.0).compute().to_numpy()
                values_filter = values!=-999.0
                values = values[values_filter]
                weights = weights[values_filter]
                print(f"weights.shape: {weights[weights>0].shape}")
                np_hist, _ = np.histogram(values, bins=binning, weights = weights)
                np_hist_w2, _ = np.histogram(values, bins=binning, weights = weights*weights)
               
                
                if process in group_data_processes:
                    print("data activated")
                    group_data_hists.append(np_hist)
                    print(f"np_hist: {np_hist}")
                    print(f"np_hist.dtype: {np_hist.dtype}")
                    group_data_hists_w2.append(np_hist_w2)
                #-------------------------------------------------------
                elif process in group_DY_processes:
                    print("DY activated")
                    group_DY_hists.append(np_hist)
                    group_DY_hists_w2.append(np_hist_w2)
                #-------------------------------------------------------
                elif process in group_Top_processes:
                    print("top activated")
                    group_Top_hists.append(np_hist)
                    group_Top_hists_w2.append(np_hist_w2)
                #-------------------------------------------------------
                elif process in group_Ewk_processes:
                    print("Ewk activated")
                    group_Ewk_hists.append(np_hist)
                    group_Ewk_hists_w2.append(np_hist_w2)
                #-------------------------------------------------------
                elif process in group_VV_processes:
                    print("VV activated")
                    # var_hist_VV = ROOT.TH1F( var+'_hist_VV', var, len(binning)-1, min(binning), max(binning))
                    # for idx in range (len(np_hist)): # paste the np histogram values to root histogram
                    #     var_hist_VV.SetBinContent(1+idx, np_hist[idx])
                    group_VV_hists.append(np_hist)
                    group_VV_hists_w2.append(np_hist_w2)
                #-------------------------------------------------------
                elif process in group_ggH_processes:
                    print("ggH activated")
                    group_ggH_hists.append(np_hist)
                #-------------------------------------------------------
                elif process in group_VBF_processes:
                    print("VBF activated")
                    group_VBF_hists.append(np_hist)
                #-------------------------------------------------------
                else: # put into "other" bkg group
                    # if "dy_M-50" in process:
                    #     # print("dy_M-50 activated")
                    #     continue
                    print("other activated")
                    group_other_hists.append(np_hist)
                    group_otherhists_w2.append(np_hist_w2)


                
                
            all_MC_hist_list = []
            all_MC_hist_list_w2 = []
            groups = []
            if len(group_DY_hists) > 0:
                DY_hist_stacked = np.sum(np.asarray(group_DY_hists), axis=0)
                all_MC_hist_list.append(DY_hist_stacked)
                # add w2 for error calculation
                DY_hist_stacked_w2 = np.sum(np.asarray(group_DY_hists_w2), axis=0)
                all_MC_hist_list_w2.append(DY_hist_stacked_w2)
                groups.append("DY")
            #----------------------------------------------
            if len(group_Top_hists) > 0:
                Top_hist_stacked = np.sum(np.asarray(group_Top_hists), axis=0)
                all_MC_hist_list.append(Top_hist_stacked)
                # add w2 for error calculation
                Top_hist_stacked_w2 = np.sum(np.asarray(group_Top_hists_w2), axis=0)
                all_MC_hist_list_w2.append(Top_hist_stacked_w2)
                groups.append("Top")
            #----------------------------------------------
            if len(group_Ewk_hists) > 0:
                Ewk_hist_stacked = np.sum(np.asarray(group_Ewk_hists), axis=0)
                all_MC_hist_list.append(Ewk_hist_stacked)
                # add w2 for error calculation
                Ewk_hist_stacked_w2 = np.sum(np.asarray(group_Ewk_hists_w2), axis=0)
                all_MC_hist_list_w2.append(Ewk_hist_stacked_w2)
                groups.append("Ewk")
            #----------------------------------------------
            if len(group_VV_hists) > 0:
                VV_hist_stacked = np.sum(np.asarray(group_VV_hists), axis=0)
                all_MC_hist_list.append(VV_hist_stacked)
                # add w2 for error calculation
                VV_hist_stacked_w2 = np.sum(np.asarray(group_VV_hists_w2), axis=0)
                all_MC_hist_list_w2.append(VV_hist_stacked_w2)
                groups.append("VV")
            #----------------------------------------------
            if len(group_other_hists) > 0:
                other_hist_stacked = np.sum(np.asarray(group_other_hists), axis=0)
                all_MC_hist_list.append(other_hist_stacked)
                # add w2 for error calculation
                other_hist_stacked_w2 = np.sum(np.asarray(group_other_hists_w2), axis=0)
                all_MC_hist_list_w2.append(other_hist_stacked_w2)
                groups.append("other")
            #----------------------------------------------     
                
                   
                
            # colours = hep.style.cms.cmap_petroff[0:3]
            colours = hep.style.cms.cmap_petroff[0:2]
            # print(f"colours: {colours}")
            # print(f"labels: {labels}")
            if not args.no_ratio:
                fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            else: # skip ratio
                fig, ax_main = plt.subplots()
            
            fig.subplots_adjust(hspace=0.1)
            # obtain fraction weight, this should be the same for all processes and rows
            # fraction_weight = 1/events.fraction[0].compute() # directly apply these to np hists
            fraction_weight = 1
            # print(f"fraction_weight: {(fraction_weight)}")
            # print(f"all_MC_hist_list: {(all_MC_hist_list)}")
            #------------------------------------------
            mc_sum_histogram = np.sum(np.asarray(all_MC_hist_list), axis=0) # to be used in ratio plot later
            group_color_map = {
                "DY" : "Orange",
                "Top" : "Green",
                "Ewk" : "Magenta",
                "VV" : "Azure",
                "other" : "Gray"
            }
            colours = [group_color_map[group] for group in groups]
            if len(all_MC_hist_list) > 0:
                hep.histplot(all_MC_hist_list, bins=binning, 
                             stack=True, histtype='fill', 
                             label=groups, 
                             sort='label_r', 
                             color=colours, 
                             ax=ax_main)

            if len(group_ggH_hists) > 0: # there should be only one element or be empty
                # hist_ggh = group_ggH_hists[0]*fraction_weight
                hist_ggh = group_ggH_hists[0]
                hep.histplot(hist_ggh, bins=binning, 
                             histtype='step', 
                             label="ggH", 
                             sort='label_r', 
                             # color =  hep.style.cms.cmap_petroff[5],
                             color =  "black",
                             # density=plot_settings[plot_name].get("density"), 
                             ax=ax_main)
            if len(group_VBF_hists) > 0: # there should be only one element or be empty
                # hist_vbf = group_VBF_hists[0]*fraction_weight
                hist_vbf = group_VBF_hists[0]
                hep.histplot(hist_vbf, bins=binning, 
                             histtype='step', 
                             label="VBF", 
                             sort='label_r', 
                             # color =  hep.style.cms.cmap_petroff[4],
                             color = "red",
                             # density=plot_settings[plot_name].get("density"), 
                             ax=ax_main)
            
            
            # data_rel_err = np.zeros_like(data_hist)
            # data_rel_err[data_hist>0] = np.sqrt(data_hist)**(-1) # poisson err / value == inverse sqrt()
            #apply fraction weight to data hist and yerr
            data_hist = np.sum(np.asarray(group_data_hists), axis=0)
            print(f"data_hist: {data_hist}")
            print(f"data_hist.dtype: {data_hist.dtype}")
            data_err = np.sqrt(np.sum(np.asarray(group_data_hists_w2), axis=0)) # sqrt of sum of squares of weights
            hep.histplot(data_hist, xerr=True, yerr=data_err,
                         bins=binning, stack=False, histtype='errorbar', color='black', 
                         label='Data', ax=ax_main)
            ax_main.set_ylabel(plot_settings[var].get("ylabel"))
            if not args.linear_scale:
                ax_main.set_yscale('log')
                ax_main.set_ylim(0.01, 1e9)
            ax_main.legend(loc="upper right")
            
            if not args.no_ratio:
                mc_yerr = np.sqrt(np.sum(np.asarray(all_MC_hist_list_w2), axis=0)) # sqrt of sum of squares of weights

                ratio_hist = np.zeros_like(data_hist)
                ratio_hist[mc_sum_histogram>0] = data_hist[mc_sum_histogram>0]/  mc_sum_histogram[mc_sum_histogram>0]
                # add rel unc of data and mc by quadrature
                rel_unc_ratio = np.sqrt((mc_yerr/mc_sum_histogram)**2 + (data_err/data_hist)**2)
                ratio_err = rel_unc_ratio*ratio_hist
                
                hep.histplot(ratio_hist, 
                             bins=binning, histtype='errorbar', yerr=ratio_err, 
                             color='black', label='Ratio', ax=ax_ratio)
                ax_ratio.axhline(1, color='gray', linestyle='--')
                ax_ratio.set_xlabel(plot_settings[var].get("xlabel"))
                ax_ratio.set_ylabel('Data / MC')
                ax_ratio.set_xlim(binning[0], binning[-1])
                # ax_ratio.set_ylim(0.6, 1.4)
                ax_ratio.set_ylim(0.5,1.5) 
            else:  
                ax_main.set_xlabel(plot_settings[var].get("xlabel"))
            # Decorating with CMS label
            if args.lumi == '':
                hep.cms.label(data=True, loc=0, label=status, com=13, ax=ax_main)
            else:
                hep.cms.label(data=True, loc=0, label=status, com=13, lumi=args.lumi, ax=ax_main)

            
            # Saving with special name
            full_save_path = args.save_path+f"/{args.year}/mplhep/Reg_{args.region}"
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
            plt.savefig(f"{full_save_path}/{var}.pdf")
            plt.clf()
            print(f"figure saved in {full_save_path}/{var}.pdf")
            # record time it took
            var_elapsed = round(time.time() - var_step, 3)
            print(f"Finished processing {var} in {var_elapsed} s.")
    time_elapsed = round(time.time() - time_step, 3)
    print(f"Finished in {time_elapsed} s.")

