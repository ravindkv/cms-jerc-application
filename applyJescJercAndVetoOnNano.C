// Self-contained ROOT macro to apply CMS JECs and print jet/MET at each correction step

#if defined(__CLING__)
#pragma cling add_include_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/include")
#pragma cling add_library_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/lib")
#pragma cling load("libcorrectionlib.so")
#endif

#include <TSystem.h>
#include <TInterpreter.h>
#include <TFile.h>
#include <TROOT.h>
#include <TChain.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <correction.h>

//------------------------------------------------------------------------------
// ReadConfig: load JSON config and retrieve values
//------------------------------------------------------------------------------
class ReadConfig {
public:
    explicit ReadConfig(const std::string& filename)
        : filename_(filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config: " + filename);
        }
        file >> config_;
    }

    template<typename T>
    T getValue(std::initializer_list<std::string> keys) const {
        nlohmann::json current = config_;
        for (auto& key : keys) {
            if (!current.contains(key)) {
                throw std::runtime_error("Missing JSON element: " + filename_ + " -> " + key);
            }
            current = current[key];
        }
        return current.get<T>();
    }

private:
    nlohmann::json config_;
    std::string filename_;
};

//------------------------------------------------------------------------------
// NanoTree: wrapper for a TChain to read NanoAOD
//------------------------------------------------------------------------------
class NanoTree {
public:
    NanoTree()
        : fCurrent_(-1), fChain_(std::make_unique<TChain>("Events")) {}
    ~NanoTree() = default;

    void loadTree(const std::string& path) {
        std::cout << "==> loadTree: " << path << std::endl;
        fChain_->SetCacheSize(100LL * 1024 * 1024);
        fChain_->Add(path.c_str());
        std::cout << "Entries: " << fChain_->GetEntries() << std::endl;
        fChain_->SetBranchStatus("*", true);

        // Set branch addresses
        fChain_->SetBranchAddress("run", &run);
        fChain_->SetBranchAddress("luminosityBlock", &luminosityBlock);
        fChain_->SetBranchAddress("event", &event);
        fChain_->SetBranchAddress("fixedGridRhoFastjetAll", &Rho);
        fChain_->SetBranchAddress("ChsMET_pt", &ChsMET_pt);
        fChain_->SetBranchAddress("ChsMET_phi", &ChsMET_phi);
        fChain_->SetBranchAddress("nJet", &nJet);
        fChain_->SetBranchAddress("Jet_pt", Jet_pt);
        fChain_->SetBranchAddress("Jet_eta", Jet_eta);
        fChain_->SetBranchAddress("Jet_phi", Jet_phi);
        fChain_->SetBranchAddress("Jet_mass", Jet_mass);
        fChain_->SetBranchAddress("Jet_rawFactor", Jet_rawFactor);
        fChain_->SetBranchAddress("Jet_area", Jet_area);
        fChain_->SetBranchAddress("Jet_jetId", Jet_jetId);
        fChain_->SetBranchAddress("Jet_genJetIdx", Jet_genJetIdx);
        fChain_->SetBranchAddress("nGenJet", &nGenJet);
        fChain_->SetBranchAddress("GenJet_pt", GenJet_pt);
        fChain_->SetBranchAddress("GenJet_eta", GenJet_eta);
        fChain_->SetBranchAddress("GenJet_phi", GenJet_phi);
        fChain_->SetBranchAddress("Pileup_nTrueInt", &Pileup_nTrueInt);
        fChain_->SetBranchAddress("genWeight", &genWeight);
    }

    Long64_t getEntries() const { return fChain_->GetEntries(); }
    Long64_t loadEntry(Long64_t i) {
        Long64_t centry = fChain_->LoadTree(i);
        if (centry < 0) return centry;
        if (fChain_->GetTreeNumber() != fCurrent_) {
            fCurrent_ = fChain_->GetTreeNumber();
        }
        return centry;
    }
    TChain* getChain() const { return fChain_.get(); }

    // Tree branches
    UInt_t   run{};
    UInt_t   luminosityBlock{};
    ULong64_t event{};
    Float_t  Rho{};
    Float_t  ChsMET_pt{};
    Float_t  ChsMET_phi{};
    UInt_t   nJet{};
    Float_t  Jet_pt[200]{};
    Float_t  Jet_eta[200]{};
    Float_t  Jet_phi[200]{};
    Float_t  Jet_mass[200]{};
    Float_t  Jet_rawFactor[200]{};
    Float_t  Jet_area[200]{};
    Int_t    Jet_jetId[200]{};
    Int_t    Jet_genJetIdx[200]{};
    UInt_t   nGenJet{};
    Float_t  GenJet_pt[200]{};
    Float_t  GenJet_eta[200]{};
    Float_t  GenJet_phi[200]{};
    Float_t  Pileup_nTrueInt{};
    Float_t  genWeight{};

private:
    Long_t fCurrent_;
    std::unique_ptr<TChain> fChain_;
};

//------------------------------------------------------------------------------
// ScaleJetMet: apply JECs/JER and print values at each step
//------------------------------------------------------------------------------
class ScaleJetMet {
public:
    enum class CorrectionLevel { None, L1Rc, L2Rel, L2ResL3Res };

    ScaleJetMet()
      : isMC_(true), isData_(false), randomGen_(0)
    {
        loadConfig_();
        loadRefs_();
    }

    void applyCorrections(std::shared_ptr<NanoTree>& nanoT, CorrectionLevel level) {
        // --- MET from NanoAOD
        TLorentzVector met_nano;
        met_nano.SetPtEtaPhiM(nanoT->ChsMET_pt, 0., nanoT->ChsMET_phi, 0.);
        std::cout << "[MET Nano]    Pt = " << met_nano.Pt()
                  << "  Phi = " << met_nano.Phi() << "\n";

        // working MET copy
        TLorentzVector met = met_nano;

        for (UInt_t i = 0; i < nanoT->nJet; ++i) {
            if (nanoT->Jet_pt[i] < 15 || std::abs(nanoT->Jet_eta[i]) > 5.2) continue;

            // --- Nano
            TLorentzVector p4;
            p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                            nanoT->Jet_eta[i],
                            nanoT->Jet_phi[i],
                            nanoT->Jet_mass[i]);
            met += p4;
            std::cout << " \nJet["<<i<<"] Nano    Pt="<<p4.Pt()
                          <<"  M="<<p4.M()<<"\n";

            // --- Raw
            float rawScale = 1.0f - nanoT->Jet_rawFactor[i];
            nanoT->Jet_pt[i]   *= rawScale;
            nanoT->Jet_mass[i] *= rawScale;
            p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                            nanoT->Jet_eta[i],
                            nanoT->Jet_phi[i],
                            nanoT->Jet_mass[i]);
            std::cout << " Jet["<<i<<"] Raw     Pt="<<p4.Pt()
                      <<"  M="<<p4.M()<<"\n";

            // --- L1RC
            if (level >= CorrectionLevel::L1Rc) {
                double c1 = corrL1_->evaluate({nanoT->Jet_area[i],
                                               nanoT->Jet_eta[i],
                                               nanoT->Jet_pt[i],
                                               nanoT->Rho});
                nanoT->Jet_pt[i]   *= c1;
                nanoT->Jet_mass[i] *= c1;
                p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                                nanoT->Jet_eta[i],
                                nanoT->Jet_phi[i],
                                nanoT->Jet_mass[i]);
                std::cout << " Jet["<<i<<"] L1Rc    Pt="<<p4.Pt()
                          <<"  M="<<p4.M()<<"\n";
            }

            // --- L2Rel
            if (level >= CorrectionLevel::L2Rel) {
                double c2 = corrL2_->evaluate({nanoT->Jet_eta[i],
                                               nanoT->Jet_pt[i]});
                nanoT->Jet_pt[i]   *= c2;
                nanoT->Jet_mass[i] *= c2;
                p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                                nanoT->Jet_eta[i],
                                nanoT->Jet_phi[i],
                                nanoT->Jet_mass[i]);
                std::cout << " Jet["<<i<<"] L2Rel   Pt="<<p4.Pt()
                          <<"  M="<<p4.M()<<"\n";
            }

            // --- L2ResL3Res (data only)
            if (isData_ && level >= CorrectionLevel::L2ResL3Res) {
                double cR = corrL2ResL3Res_->evaluate({nanoT->Jet_eta[i],
                                                       nanoT->Jet_pt[i]});
                nanoT->Jet_pt[i]   *= cR;
                nanoT->Jet_mass[i] *= cR;
                p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                                nanoT->Jet_eta[i],
                                nanoT->Jet_phi[i],
                                nanoT->Jet_mass[i]);
                std::cout << " Jet["<<i<<"] L2ResL3 Pt="<<p4.Pt()
                          <<"  M="<<p4.M()<<"\n";
            }

            // --- JER smearing (MC only)
            if (!isData_) {
                double reso = corrJerReso_->evaluate({nanoT->Jet_eta[i],
                                                      nanoT->Jet_pt[i],
                                                      nanoT->Rho});
                double sf   = corrJerSf_->evaluate({nanoT->Jet_eta[i],
                                                    std::string("nom")});
                randomGen_.SetSeed(nanoT->event + nanoT->run + nanoT->luminosityBlock);
                double smear = std::max(0.0, 1 + randomGen_.Gaus(0, reso)
                                             * std::sqrt(std::max(sf*sf - 1.0, 0.0)));
                nanoT->Jet_pt[i]   *= smear;
                nanoT->Jet_mass[i] *= smear;
                p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                                nanoT->Jet_eta[i],
                                nanoT->Jet_phi[i],
                                nanoT->Jet_mass[i]);
                std::cout << " Jet["<<i<<"] Jer     Pt="<<p4.Pt()
                          <<"  M="<<p4.M()<<"\n";
            }

            // --- Final Corr
            p4.SetPtEtaPhiM(nanoT->Jet_pt[i],
                            nanoT->Jet_eta[i],
                            nanoT->Jet_phi[i],
                            nanoT->Jet_mass[i]);
            std::cout << " Jet["<<i<<"] Corr    Pt="<<p4.Pt()
                      <<"  M="<<p4.M()<<"\n";

            met -= p4;  // subtract final-corrected jet for MET
        }

        // --- print corrected MET
        std::cout << "[MET Corr]    Pt = " << met.Pt()
                  << "  Phi = " << met.Phi() << "\n\n";

        // update tree variables
        nanoT->ChsMET_pt  = met.Pt();
        nanoT->ChsMET_phi = met.Phi();
    }

private:
    void loadConfig_() {
        ReadConfig cfg("JescJercAndVetoTagName.json");
        auto year = std::string("2018");
        jercPath_      = cfg.getValue<std::string>({year, "jercJsonPath"});
        nameL1_        = cfg.getValue<std::string>({year, "jetL1FastJetName"});
        nameL2_        = cfg.getValue<std::string>({year, "jetL2RelativeName"});
        nameL2ResL3Res_= cfg.getValue<std::string>({year, "jetL2L3ResidualName"});
        nameJR_        = cfg.getValue<std::string>({year, "JerResoName"});
        nameJS_        = cfg.getValue<std::string>({year, "JerSfName"});
    }
    void loadRefs_() {
        auto cs = correction::CorrectionSet::from_file(jercPath_);
        corrL1_         = cs->at(nameL1_);
        corrL2_         = cs->at(nameL2_);
        corrL2ResL3Res_ = cs->at(nameL2ResL3Res_);
        corrJerReso_    = cs->at(nameJR_);
        corrJerSf_      = cs->at(nameJS_);
    }

    bool isMC_, isData_;
    TRandom3 randomGen_;

    std::string jercPath_, nameL1_, nameL2_, nameL2ResL3Res_, nameJR_, nameJS_;
    correction::Correction::Ref corrL1_, corrL2_, corrL2ResL3Res_, corrJerReso_, corrJerSf_;
};

//------------------------------------------------------------------------------
// Macro entry point
//------------------------------------------------------------------------------
void applyJescJercAndVetoOnNano(const char* inFile="NanoAod.root") {
    auto nanoT = std::make_shared<NanoTree>();
    nanoT->loadTree(inFile);

    auto scale = std::make_shared<ScaleJetMet>();
    Long64_t n = nanoT->getEntries();
    std::cout << "Events: " << n << "\n";

    for (Long64_t i = 0; i < std::min(n, (Long64_t)10); ++i) {
        Long64_t c = nanoT->loadEntry(i);
        if (c < 0) break;
        nanoT->getChain()->GetTree()->GetEntry(c);

        scale->applyCorrections(nanoT, ScaleJetMet::CorrectionLevel::L2ResL3Res);
        std::cout << "================== End of Event " << i << " ==================\n\n";
    }
}

