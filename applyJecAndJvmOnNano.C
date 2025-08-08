// applyJECandJVMonNano.C
// Flattened ROOT macro: apply JEC/JER using JSON config and NanoAOD input.
// Supports MC vs Data with era selection. No unnecessary class indirection.

#if defined(__CLING__)
#pragma cling add_include_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/include")
#pragma cling add_library_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/lib")
#pragma cling load("libcorrectionlib.so")
#endif

#include <TChain.h>
#include <TLorentzVector.h>
#include <TRandom3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TDirectory.h>


#include <nlohmann/json.hpp>
#include <correction.h>

// ---------------------------
// JSON / Tag helpers
// ---------------------------
struct Tags {
    std::string jercJsonPath;
    std::string jetL1FastJetName;
    std::string jetL2RelativeName;
    std::string jetL3AbsoluteName;
    std::string jetL2ResidualName;
    std::string jetL2L3ResidualName;
    std::string jerResoName;
    std::string jerSfName;
};

nlohmann::json loadJsonConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open JSON config: " + filename);
    }
    nlohmann::json j;
    f >> j;
    return j;
}

std::string getRequiredString(const nlohmann::json& obj, const std::string& key) {
    if (!obj.contains(key)) {
        throw std::runtime_error("Missing required key in JSON: " + key);
    }
    return obj.at(key).get<std::string>();
}

Tags resolveTags(const nlohmann::json& baseJson,
                 const std::string& year,
                 bool isData,
                 const std::optional<std::string>& era) 
{
    if (!baseJson.contains(year)) {
        throw std::runtime_error("Year key not found in JSON: " + year);
    }
    const auto& yearObj = baseJson.at(year);
    Tags t;

    t.jercJsonPath          = getRequiredString(yearObj, "jercJsonPath");
    t.jetL1FastJetName      = getRequiredString(yearObj, "jetL1FastJetName");
    t.jetL2RelativeName     = getRequiredString(yearObj, "jetL2RelativeName");
    t.jetL3AbsoluteName     = getRequiredString(yearObj, "jetL3AbsoluteName");
    t.jetL2ResidualName     = getRequiredString(yearObj, "jetL2ResidualName");
    t.jetL2L3ResidualName   = getRequiredString(yearObj, "jetL2L3ResidualName");
    t.jerResoName           = getRequiredString(yearObj, "JerResoName");
    t.jerSfName             = getRequiredString(yearObj, "JerSfName");

    if (isData) {
        if (!yearObj.contains("data")) {
            throw std::runtime_error("Requested data but no 'data' section for year: " + year);
        }
        if (!era.has_value()) {
            throw std::runtime_error("Data requested but no era provided for year: " + year);
        }
        const std::string& eraKey = era.value();
        const auto& dataObj = yearObj.at("data");
        if (!dataObj.contains(eraKey)) {
            throw std::runtime_error("Era key not found under data for year " + year + ": " + eraKey);
        }
        const auto& eraObj = dataObj.at(eraKey);
        t.jetL1FastJetName    = getRequiredString(eraObj, "jetL1FastJetName");
        t.jetL2RelativeName   = getRequiredString(eraObj, "jetL2RelativeName");
        t.jetL3AbsoluteName   = getRequiredString(eraObj, "jetL3AbsoluteName");
        t.jetL2ResidualName   = getRequiredString(eraObj, "jetL2ResidualName");
        t.jetL2L3ResidualName = getRequiredString(eraObj, "jetL2L3ResidualName");
        // JER names remain from MC unless JSON extended
    }

    return t;
}

using SystSetMap = std::map<std::string, std::vector<std::string>>;

SystSetMap resolveSystematics(const nlohmann::json& baseJson, const std::string& year) {
    SystSetMap out;
    if (!baseJson.contains(year)) return out; // empty if missing

    const auto& yearObj = baseJson.at(year);
    if (!yearObj.contains("systematics")) return out;

    const auto& systs = yearObj.at("systematics");
    for (auto it = systs.begin(); it != systs.end(); ++it) {
        const std::string setName = it.key();
        if (!it.value().is_array()) continue;
        std::vector<std::string> tags = it.value().get<std::vector<std::string>>();
        out[setName] = std::move(tags);
    }
    return out;
}


// ---------------------------
// NanoAOD flat branches
// ---------------------------
struct NanoBranches {
    UInt_t    run{};
    UInt_t    luminosityBlock{};
    ULong64_t event{};
    Float_t   Rho{};
    Float_t   ChsMET_pt{};
    Float_t   ChsMET_phi{};
    UInt_t    nJet{};
    Float_t   Jet_pt[200]{};
    Float_t   Jet_eta[200]{};
    Float_t   Jet_phi[200]{};
    Float_t   Jet_mass[200]{};
    Float_t   Jet_rawFactor[200]{};
    Float_t   Jet_area[200]{};
    Int_t     Jet_jetId[200]{};
    Int_t     Jet_genJetIdx[200]{};
    UInt_t    nGenJet{};
    Float_t   GenJet_pt[200]{};
    Float_t   GenJet_eta[200]{};
    Float_t   GenJet_phi[200]{};
    Float_t   Pileup_nTrueInt{};
    Float_t   genWeight{};
};

void setupNanoBranches(TChain* chain, NanoBranches& nb) {
    chain->SetBranchStatus("*", true);
    chain->SetBranchAddress("run", &nb.run);
    chain->SetBranchAddress("luminosityBlock", &nb.luminosityBlock);
    chain->SetBranchAddress("event", &nb.event);
    chain->SetBranchAddress("fixedGridRhoFastjetAll", &nb.Rho);
    chain->SetBranchAddress("ChsMET_pt", &nb.ChsMET_pt);
    chain->SetBranchAddress("ChsMET_phi", &nb.ChsMET_phi);
    chain->SetBranchAddress("nJet", &nb.nJet);
    chain->SetBranchAddress("Jet_pt", nb.Jet_pt);
    chain->SetBranchAddress("Jet_eta", nb.Jet_eta);
    chain->SetBranchAddress("Jet_phi", nb.Jet_phi);
    chain->SetBranchAddress("Jet_mass", nb.Jet_mass);
    chain->SetBranchAddress("Jet_rawFactor", nb.Jet_rawFactor);
    chain->SetBranchAddress("Jet_area", nb.Jet_area);
    chain->SetBranchAddress("Jet_jetId", nb.Jet_jetId);
    chain->SetBranchAddress("Jet_genJetIdx", nb.Jet_genJetIdx);
    chain->SetBranchAddress("nGenJet", &nb.nGenJet);
    chain->SetBranchAddress("GenJet_pt", nb.GenJet_pt);
    chain->SetBranchAddress("GenJet_eta", nb.GenJet_eta);
    chain->SetBranchAddress("GenJet_phi", nb.GenJet_phi);
    chain->SetBranchAddress("Pileup_nTrueInt", &nb.Pileup_nTrueInt);
    chain->SetBranchAddress("genWeight", &nb.genWeight);
}

correction::Correction::Ref safeGet(const std::shared_ptr<correction::CorrectionSet>& cs,
                                   const std::string& name) 
{
    try {
        return cs->at(name);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to retrieve correction \"" + name + "\": " + e.what());
    }
}


// ---------------------------
// Correction references and application (flattened)
// ---------------------------
struct CorrectionRefs {
    std::shared_ptr<correction::CorrectionSet> cs;
    correction::Correction::Ref corrL1;
    correction::Correction::Ref corrL2;
    correction::Correction::Ref corrL2ResL3Res;
    correction::Correction::Ref corrJerReso;
    correction::Correction::Ref corrJerSf;
    TRandom3 randomGen;

    CorrectionRefs(const Tags& tags)
        : randomGen(0)
    {
        static std::map<std::string, std::shared_ptr<correction::CorrectionSet>> cache;

        if (cache.count(tags.jercJsonPath)) {
            cs = cache.at(tags.jercJsonPath);
        } else {
            cs = correction::CorrectionSet::from_file(tags.jercJsonPath);
            cache[tags.jercJsonPath] = cs;
        }

        corrL1         = safeGet(cs, tags.jetL1FastJetName);
        corrL2         = safeGet(cs, tags.jetL2RelativeName);
        corrL2ResL3Res = safeGet(cs, tags.jetL2L3ResidualName);
        corrJerReso    = safeGet(cs, tags.jerResoName);
        corrJerSf      = safeGet(cs, tags.jerSfName);

    }
};

// Build a TLorentzVector from NanoAOD jet branches
TLorentzVector makeJetP4(const NanoBranches& nb, UInt_t idx) {
    TLorentzVector p4;
    p4.SetPtEtaPhiM(nb.Jet_pt[idx], nb.Jet_eta[idx], nb.Jet_phi[idx], nb.Jet_mass[idx]);
    return p4;
}

// Apply a multiplicative scale factor to both the stored branch values
// and the working four-vector.
void applyScale(NanoBranches& nb, UInt_t idx, TLorentzVector& p4, double scale) {
    nb.Jet_pt[idx]   *= scale;
    nb.Jet_mass[idx] *= scale;
    p4.SetPtEtaPhiM(nb.Jet_pt[idx], nb.Jet_eta[idx], nb.Jet_phi[idx], nb.Jet_mass[idx]);
}

// Correct a single jet and update MET. All verbose printing is optional
// to keep the nominal correction loop concise.
void correctJet(NanoBranches& nb,
                CorrectionRefs& refs,
                bool isData,
                UInt_t idx,
                TLorentzVector& met,
                bool verbose=false) {
    TLorentzVector p4 = makeJetP4(nb, idx);
    met += p4;
    if (verbose) {
        std::cout << "\nJet[" << idx << "] Nano    Pt=" << p4.Pt()
                  << "  M=" << p4.M() << "\n";
    }

    // Raw correction
    applyScale(nb, idx, p4, 1.0f - nb.Jet_rawFactor[idx]);
    if (verbose) {
        std::cout << " Jet[" << idx << "] Raw     Pt=" << p4.Pt()
                  << "  M=" << p4.M() << "\n";
    }

    // L1 fastjet
    double c1 = refs.corrL1->evaluate({nb.Jet_area[idx],
                                       nb.Jet_eta[idx],
                                       nb.Jet_pt[idx],
                                       nb.Rho});
    applyScale(nb, idx, p4, c1);
    if (verbose) {
        std::cout << " Jet[" << idx << "] L1Rc    Pt=" << p4.Pt()
                  << "  M=" << p4.M() << "\n";
    }

    // L2 relative
    double c2 = refs.corrL2->evaluate({nb.Jet_eta[idx], nb.Jet_pt[idx]});
    applyScale(nb, idx, p4, c2);
    if (verbose) {
        std::cout << " Jet[" << idx << "] L2Rel   Pt=" << p4.Pt()
                  << "  M=" << p4.M() << "\n";
    }

    // Residual for data
    if (isData) {
        double cR = refs.corrL2ResL3Res->evaluate({nb.Jet_eta[idx], nb.Jet_pt[idx]});
        applyScale(nb, idx, p4, cR);
        if (verbose) {
            std::cout << " Jet[" << idx << "] L2ResL3 Pt=" << p4.Pt()
                      << "  M=" << p4.M() << "\n";
        }
    }

    // JER smearing for MC
    if (!isData) {
        double reso = refs.corrJerReso->evaluate({nb.Jet_eta[idx], nb.Jet_pt[idx], nb.Rho});
        double sf   = refs.corrJerSf->evaluate({nb.Jet_eta[idx], std::string("nom")});
        refs.randomGen.SetSeed(static_cast<UInt_t>(nb.event + nb.run + nb.luminosityBlock));
        double smear = std::max(0.0, 1 + refs.randomGen.Gaus(0, reso)
                                     * std::sqrt(std::max(sf * sf - 1.0, 0.0)));
        applyScale(nb, idx, p4, smear);
        if (verbose) {
            std::cout << " Jet[" << idx << "] Jer     Pt=" << p4.Pt()
                      << "  M=" << p4.M() << "\n";
        }
    }

    // Final jet after all corrections
    if (verbose) {
        std::cout << " Jet[" << idx << "] Corr    Pt=" << p4.Pt()
                  << "  M=" << p4.M() << "\n";
    }

    met -= p4; // subtract corrected jet
}

void applyJecNominal(NanoBranches& nb, CorrectionRefs& refs, bool isData) {
    TLorentzVector met;
    met.SetPtEtaPhiM(nb.ChsMET_pt, 0., nb.ChsMET_phi, 0.);
    std::cout << "[MET Nano]    Pt = " << met.Pt()
              << "  Phi = " << met.Phi() << "\n";

    for (UInt_t i = 0; i < nb.nJet; ++i) {
        if (nb.Jet_pt[i] < 15 || std::abs(nb.Jet_eta[i]) > 5.2) continue;
        correctJet(nb, refs, isData, i, met);
    }

    std::cout << "[MET Corr]    Pt = " << met.Pt()
              << "  Phi = " << met.Phi() << "\n\n";
    nb.ChsMET_pt  = met.Pt();
    nb.ChsMET_phi = met.Phi();
}

void applySystematicShift(NanoBranches& nb,
                          CorrectionRefs& refs,
                          const std::string& systName, const std::string& systVariation)
{
    // Example: fetch the extra correction for this systematic and apply like a scale
    // This assumes the syst correction takes (eta, pt) or similar; adapt inputs as needed.
    correction::Correction::Ref systCorr = safeGet(refs.cs, systName);

    for (UInt_t i = 0; i < nb.nJet; ++i) {
        if (nb.Jet_pt[i] < 15 || std::abs(nb.Jet_eta[i]) > 5.2) continue;

        TLorentzVector p4 = makeJetP4(nb, i);

        // Evaluate the systematic scale (example signature; replace with correct one)
        double scale = 1.0;
        try {
            scale = systCorr->evaluate({nb.Jet_eta[i], nb.Jet_pt[i]});
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to evaluate systematic " << systName
                      << " for jet " << i << ": " << e.what() << "\n";
        }

        double shift = (systVariation == "Up") ? (1 + scale) : (1 - scale);
        applyScale(nb, i, p4, shift);
        std::cout << " Jet[" << i << "] Syst    Pt=" << p4.Pt()
                  << "  systVar=" << systVariation
                  << "  scale=" << scale
                  << "  shift=" << shift << "\n";
    }
}


// ---------------------------
// Entry point
// ---------------------------
void processSampleWithSystematics(const std::string& inputFile,
                                  TFile& fout,
                                  const std::string& year,
                                  bool isData,
                                  const std::optional<std::string>& era="")
{
    auto baseJson = loadJsonConfig("configJecAndJvm.json");
    Tags tags = resolveTags(baseJson, year, isData, era);
    SystSetMap systSets = resolveSystematics(baseJson, year);

    TChain chain("Events");
    chain.SetCacheSize(100LL * 1024 * 1024);
    chain.Add(inputFile.c_str());

    NanoBranches nb;
    setupNanoBranches(&chain, nb);

    CorrectionRefs refs(tags);

    // Directory hierarchy
    TDirectory* yearDir = fout.mkdir(year.c_str());
    TDirectory* typeDir = nullptr;
    if (isData) typeDir = yearDir->mkdir("Data");
    else      typeDir = yearDir->mkdir("MC");

    // Nominal histogram (example: leading jet pt)
    typeDir->cd();
    TH1F* h_nominal = new TH1F("JetPt_JEC_Nominal", "Nominal Jet p_{T};p_{T} [GeV];entries", 100, 0, 500);

    // Systematic histograms: map[set][tag] -> histogram
    std::map<std::string, std::map<std::string, TH1F*>> h_systs;
    for (const auto& [setName, tagsVec] : systSets) {
        TDirectory* systSetDir = nullptr;
        systSetDir = typeDir->mkdir(setName.c_str());
        systSetDir->cd();
        std::vector<std::string> systVariation = {"Up", "Down"};
        for (const auto& systTag : tagsVec) {
            for(const auto& systVar: systVariation){
                std::string histName = "JetPt_JEC_"+ systVar+ "_"+ systTag;
                h_systs[setName][systTag] =
                new TH1F(histName.c_str(), (histName + ";p_{T} [GeV];entries").c_str(), 100, 0, 500);
            }
        }
        typeDir->cd();
    }

    Long64_t nEntries = chain.GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        Long64_t centry = chain.LoadTree(i);
        if (centry < 0) break;
        chain.GetTree()->GetEntry(centry);

        // --- Nominal ---
        applyJecNominal(nb, refs, isData);
        // Fill leading jet pt as example
        double eventWeight = 1.0;
        if (nb.nJet > 0) {
            float pt0 = nb.Jet_pt[0];
            h_nominal->Fill(pt0, eventWeight);
        }

        // Keep a copy of "Nominal" corrected branches
        NanoBranches nbNominalJec = nb;

        if(isData){
            // --- Systematics ---
            for (const auto& [setName, tagsVec] : systSets) {
                for (const auto& systTag : tagsVec) {
                    // restore Nominal
                    nb = nbNominalJec;
                    // then apply this systematic shift Up
                    applySystematicShift(nb, refs, systTag, "Up");
                    // restore Nominal
                    nb = nbNominalJec;
                    // then apply this systematic shift Down
                    applySystematicShift(nb, refs, systTag, "Down");
                    if (nb.nJet > 0) {
                        float pt0 = nb.Jet_pt[0];
                        h_systs[setName][systTag]->Fill(pt0, eventWeight);
                    }
                }//tag loop
            }//Syst loop
            // (Optionally: reset nb for next event)
        }//isMC
    }//Event loop
}


// ---------------------------
// Example batch driver
// ---------------------------
void applyJecAndJvmOnNano() {
    const std::string fInputMc = "NanoAod_MC.root";
    const std::string fInputData = "NanoAod_Data.root";

    // Prepare output file
    std::string outName = "output.root";
    TFile fout(outName.c_str(), "RECREATE");


    std::vector<std::string> mcYears = {"2017", "2018"};
    std::vector<std::pair<std::string, std::string>> dataConfigs = {
        {"2017", "Era2017B"},
        //{"2018", "Era2018A"}
    };

    for (const auto& y : mcYears) {
        //processSampleWithSystematics(fInputMc, fout, y, false);
    }

    for (const auto& [year, era] : dataConfigs) {
        processSampleWithSystematics(fInputData, fout, year, true, era);
    }

    fout.Write();
    fout.Close();
    std::cout << "Wrote output to " << outName << "\n";
}

