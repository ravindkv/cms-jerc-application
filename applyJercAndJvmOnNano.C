//applyJercAndJvmOnNano.C 

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
// NanoAOD flat branches
// ---------------------------
struct NanoTree {
    UInt_t    run{};
    UInt_t    luminosityBlock{};
    ULong64_t event{};
    Float_t   Rho{};
    Float_t   MET_pt{};
    Float_t   MET_phi{};
    Int_t     nJet{};
    Float_t   Jet_pt[200]{};
    Float_t   Jet_eta[200]{};
    Float_t   Jet_phi[200]{};
    Float_t   Jet_mass[200]{};
    Float_t   Jet_rawFactor[200]{};
    Float_t   Jet_area[200]{};
    UChar_t   Jet_jetId[200]{};
    Short_t   Jet_genJetIdx[200]{};
    Float_t   Jet_neEmEF[200]{};
    Float_t   Jet_chEmEF[200]{};
    Int_t     nFatJet{};
    Float_t   FatJet_pt[200]{};
    Float_t   FatJet_eta[200]{};
    Float_t   FatJet_phi[200]{};
    Float_t   FatJet_mass[200]{};
    Float_t   FatJet_rawFactor[200]{};
    Float_t   FatJet_area[200]{};
    UChar_t   FatJet_jetId[200]{};
    Short_t   FatJet_genJetAK8Idx[200]{};
    Int_t     nGenJet{};
    Float_t   GenJet_pt[200]{};
    Float_t   GenJet_eta[200]{};
    Float_t   GenJet_phi[200]{};
    Int_t     nGenJetAK8{};
    Float_t   GenJetAK8_pt[200]{};
    Float_t   GenJetAK8_eta[200]{};
    Float_t   GenJetAK8_phi[200]{};
    Int_t nMuon{};
    Float_t Muon_pt[100]{};
    Float_t Muon_eta[100]{};
    Float_t Muon_phi[100]{};
    Float_t Muon_mass[100]{};
    Bool_t Muon_tightId[100]{};
};

void setupNanoBranches(TChain* chain, NanoTree& nanoT, bool isData) {
    chain->SetBranchStatus("*", true);
    chain->SetBranchAddress("run", &nanoT.run);
    chain->SetBranchAddress("luminosityBlock", &nanoT.luminosityBlock);
    chain->SetBranchAddress("event", &nanoT.event);
    chain->SetBranchAddress("Rho_fixedGridRhoFastjetAll", &nanoT.Rho);
    chain->SetBranchAddress("MET_pt", &nanoT.MET_pt);
    chain->SetBranchAddress("MET_phi", &nanoT.MET_phi);
    chain->SetBranchAddress("nJet", &nanoT.nJet);
    chain->SetBranchAddress("Jet_pt", &nanoT.Jet_pt);
    chain->SetBranchAddress("Jet_eta", &nanoT.Jet_eta);
    chain->SetBranchAddress("Jet_phi", &nanoT.Jet_phi);
    chain->SetBranchAddress("Jet_mass", &nanoT.Jet_mass);
    chain->SetBranchAddress("Jet_rawFactor", &nanoT.Jet_rawFactor);
    chain->SetBranchAddress("Jet_area", &nanoT.Jet_area);
    chain->SetBranchAddress("Jet_jetId", &nanoT.Jet_jetId);
	chain->SetBranchAddress("Jet_chEmEF"  , &nanoT.Jet_chEmEF);
	chain->SetBranchAddress("Jet_neEmEF"  , &nanoT.Jet_neEmEF);
    chain->SetBranchAddress("nFatJet", &nanoT.nFatJet);
    chain->SetBranchAddress("FatJet_pt", &nanoT.FatJet_pt);
    chain->SetBranchAddress("FatJet_eta", &nanoT.FatJet_eta);
    chain->SetBranchAddress("FatJet_phi", &nanoT.FatJet_phi);
    chain->SetBranchAddress("FatJet_mass", &nanoT.FatJet_mass);
    chain->SetBranchAddress("FatJet_rawFactor", &nanoT.FatJet_rawFactor);
    chain->SetBranchAddress("FatJet_area", &nanoT.FatJet_area);
    chain->SetBranchAddress("FatJet_jetId", &nanoT.FatJet_jetId);
	chain->SetBranchAddress("nMuon", &nanoT.nMuon);
	chain->SetBranchAddress("Muon_pt", &nanoT.Muon_pt);
	chain->SetBranchAddress("Muon_eta", &nanoT.Muon_eta);
	chain->SetBranchAddress("Muon_phi", &nanoT.Muon_phi);
	chain->SetBranchAddress("Muon_mass", &nanoT.Muon_mass);
	chain->SetBranchAddress("Muon_tightId", &nanoT.Muon_tightId);
    if(!isData){//Only for MC (these branches are used in JER smearing)
        chain->SetBranchAddress("Jet_genJetIdx", &nanoT.Jet_genJetIdx);
        chain->SetBranchAddress("FatJet_genJetAK8Idx", &nanoT.FatJet_genJetAK8Idx);
        chain->SetBranchAddress("nGenJet", &nanoT.nGenJet);
        chain->SetBranchAddress("GenJet_pt", &nanoT.GenJet_pt);
        chain->SetBranchAddress("GenJet_eta", &nanoT.GenJet_eta);
        chain->SetBranchAddress("GenJet_phi", &nanoT.GenJet_phi);
        chain->SetBranchAddress("nGenJetAK8", &nanoT.nGenJetAK8);
        chain->SetBranchAddress("GenJetAK8_pt", &nanoT.GenJetAK8_pt);
        chain->SetBranchAddress("GenJetAK8_eta", &nanoT.GenJetAK8_eta);
        chain->SetBranchAddress("GenJetAK8_phi", &nanoT.GenJetAK8_phi);
    }
}

// 1) Define a Specs (specifications) class for each jet collection
struct AK4Specs {
  static Float_t  getPt(const NanoTree& nanoT, UInt_t i)       { return nanoT.Jet_pt[i]; }
  static Float_t  getEta(const NanoTree& nanoT, UInt_t i)      { return nanoT.Jet_eta[i]; }
  static Float_t  getRawFactor(const NanoTree& nanoT, UInt_t i){ return nanoT.Jet_rawFactor[i]; }
  static Float_t  getArea(const NanoTree& nanoT, UInt_t i)     { return nanoT.Jet_area[i]; }
  static TLorentzVector makeTLorentzVector(const NanoTree& nanoT, UInt_t i){
    TLorentzVector p4;
    p4.SetPtEtaPhiM(nanoT.Jet_pt[i], nanoT.Jet_eta[i], nanoT.Jet_phi[i], nanoT.Jet_mass[i]);
    return p4;
  }
  static void applyCorrection(NanoTree& nanoT, UInt_t i, double sf){
    nanoT.Jet_pt[i]   *= sf;
    nanoT.Jet_mass[i] *= sf;
  }
};

struct AK8Specs {
  static Float_t  getPt(const NanoTree& nanoT, UInt_t i)       { return nanoT.FatJet_pt[i]; }
  static Float_t  getEta(const NanoTree& nanoT, UInt_t i)      { return nanoT.FatJet_eta[i]; }
  static Float_t  getRawFactor(const NanoTree& nanoT, UInt_t i){ return nanoT.FatJet_rawFactor[i]; }
  static Float_t  getArea(const NanoTree& nanoT, UInt_t i)     { return nanoT.FatJet_area[i]; }
  static TLorentzVector makeTLorentzVector(const NanoTree& nanoT, UInt_t i){
    TLorentzVector p4;
    p4.SetPtEtaPhiM(nanoT.FatJet_pt[i], nanoT.FatJet_eta[i], nanoT.FatJet_phi[i], nanoT.FatJet_mass[i]);
    return p4;
  }
  static void applyCorrection(NanoTree& nanoT, UInt_t i, double sf){
    nanoT.FatJet_pt[i]   *= sf;
    nanoT.FatJet_mass[i] *= sf;
  }
};


// ---------------------------
// JSON / Tag helpers
// ---------------------------
struct Tags {
    std::string jercJsonPath;
    std::string tagNameL1FastJet;
    std::string tagNameL2Relative;
    std::string tagNameL3Absolute;
    std::string tagNameL2Residual;
    std::string tagNameL2L3Residual;
    std::string tagNamePtResolution;
    std::string tagNameJerScaleFactor;
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

std::string getTagName(const nlohmann::json& obj, const std::string& key) {
    if (!obj.contains(key)) {
        throw std::runtime_error("Missing required key in JSON: " + key);
    }
    return obj.at(key).get<std::string>();
}

Tags getTagNames(const nlohmann::json& baseJson,
                 const std::string& year,
                 bool isData,
                 const std::optional<std::string>& era) 
{
    if (!baseJson.contains(year)) {
        throw std::runtime_error("Year key not found in JSON: " + year);
    }
    const auto& yearObj = baseJson.at(year);
    Tags tags;

    tags.jercJsonPath         = getTagName(yearObj, "jercJsonPath");
    tags.tagNameL1FastJet     = getTagName(yearObj, "tagNameL1FastJet");
    tags.tagNameL2Relative    = getTagName(yearObj, "tagNameL2Relative");
    tags.tagNameL3Absolute    = getTagName(yearObj, "tagNameL3Absolute");
    tags.tagNameL2Residual    = getTagName(yearObj, "tagNameL2Residual");
    tags.tagNameL2L3Residual  = getTagName(yearObj, "tagNameL2L3Residual");
    tags.tagNamePtResolution  = getTagName(yearObj, "tagNamePtResolution");
    tags.tagNameJerScaleFactor= getTagName(yearObj, "tagNameJerScaleFactor");

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
        tags.tagNameL1FastJet   = getTagName(eraObj, "tagNameL1FastJet");
        tags.tagNameL2Relative  = getTagName(eraObj, "tagNameL2Relative");
        tags.tagNameL3Absolute  = getTagName(eraObj, "tagNameL3Absolute");
        tags.tagNameL2Residual  = getTagName(eraObj, "tagNameL2Residual");
        tags.tagNameL2L3Residual= getTagName(eraObj, "tagNameL2L3Residual");
    }

    return tags;
}

using SystSetMap = std::map<std::string, std::vector<std::string>>;

SystSetMap getSystTagNames(const nlohmann::json& baseJson, const std::string& year) {
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

        corrL1         = safeGet(cs, tags.tagNameL1FastJet);
        corrL2         = safeGet(cs, tags.tagNameL2Relative);
        corrL2ResL3Res = safeGet(cs, tags.tagNameL2L3Residual);
        corrJerReso    = safeGet(cs, tags.tagNamePtResolution);
        corrJerSf      = safeGet(cs, tags.tagNameJerScaleFactor);

    }
};

// 2) A single templated all nominal corrections
template<typename Specs>
void applyNominalCorrections(NanoTree& nanoT,
                       CorrectionRefs& refs,
                       bool isData,
                       const std::vector<UInt_t>& idxs,
                       bool verbose=false)
{
    for(auto idx: idxs){
        if(verbose) std::cout<<"   [Jet] index="<<idx<<"\n";
        nanoT.MET_pt += nanoT.Jet_pt[idx];//add MET to jet 
        if(verbose) std::cout<<"    default NanoAod Pt="<<nanoT.Jet_pt[idx]<<"\n";

        // raw correction
        double rawSF = 1.0 - Specs::getRawFactor(nanoT, idx);
        Specs::applyCorrection(nanoT, idx, rawSF);
        if(verbose) std::cout<<"    after undoing   Pt="<<nanoT.Jet_pt[idx]<<"\n";

        // L1
        double c1 = refs.corrL1->evaluate({ Specs::getArea(nanoT,idx),
                                            Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx),
                                            nanoT.Rho });
        Specs::applyCorrection(nanoT, idx, c1);
        if(verbose) std::cout<<"    after L1    Pt="<<nanoT.Jet_pt[idx]<<"\n";

        // L2 rel
        double c2 = refs.corrL2->evaluate({ Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx) });
        Specs::applyCorrection(nanoT, idx, c2);
        if(verbose) std::cout<<"    after L2Rel Pt="<<nanoT.Jet_pt[idx]<<"\n";

        // Residual (data only)
        if(isData){
          double cR = refs.corrL2ResL3Res->evaluate({ Specs::getEta(nanoT,idx),
                                                      Specs::getPt(nanoT,idx) });
          Specs::applyCorrection(nanoT, idx, cR);
          if(verbose) std::cout<<"    after ResL3 Pt="<<nanoT.Jet_pt[idx]<<"\n";
        }

        // JER smearing (MC only)
        if(!isData){
          double reso = refs.corrJerReso->evaluate({ Specs::getEta(nanoT,idx),
                                                      Specs::getPt(nanoT,idx),
                                                      nanoT.Rho });
          double sf   = refs.corrJerSf->evaluate({ Specs::getEta(nanoT,idx),
                                                    std::string("nom") });
          refs.randomGen.SetSeed(nanoT.event + nanoT.run + nanoT.luminosityBlock);
          double smear = std::max(0.0, 1 + refs.randomGen.Gaus(0, reso)
                                        * std::sqrt(std::max(sf*sf - 1.0, 0.0)));
          Specs::applyCorrection(nanoT, idx, smear);
          if(verbose) std::cout<<"    after JER    Pt="<<nanoT.Jet_pt[idx]<<"\n";
        }

        nanoT.MET_pt -= nanoT.Jet_pt[idx];//substract MET from jet 
    }
}

// 4) Templated systematic shifts
template<typename Specs>
void applySystematicShift(NanoTree& nanoT,
                          CorrectionRefs& refs,
                          const std::string& systName,
                          const std::string& var,
                          const std::vector<UInt_t>& idxs,
                          bool print=false
                          )
{
    auto systCorr = safeGet(refs.cs, systName);
    for(auto idx: idxs){
        nanoT.MET_pt += nanoT.Jet_pt[idx];//add MET to jet 

        if(print) std::cout<<"   Nominal corrected    Pt="<<nanoT.Jet_pt[idx]<<"\n";
        double scale = systCorr->evaluate({ Specs::getEta(nanoT,idx), Specs::getPt(nanoT,idx) });
        double shift = (var=="Up" ? 1+scale : 1-scale);
        Specs::applyCorrection(nanoT, idx, shift);
        if(print) std::cout<<"   Syst corrected    Pt="<<nanoT.Jet_pt[idx]<<"\n";

        nanoT.MET_pt += nanoT.Jet_pt[idx];//substract MET from jet 
    }
}

bool checkIfAnyJetInVetoRegion(const correction::Correction::Ref &jvmRef, std::string jvmKeyName, const NanoTree& nanoT){
    const double maxEtaInMap = 5.191;
    const double maxPhiInMap = 3.1415926;
    bool vetoEvent = false;
    for (int i = 0; i != nanoT.nJet; ++i) {
        //apply minimal selection jets
        if (std::abs(nanoT.Jet_eta[i]) > maxEtaInMap) continue;
        if (std::abs(nanoT.Jet_phi[i]) > maxPhiInMap) continue;
        if (nanoT.Jet_jetId[i] < 6 ) continue;
        if (nanoT.Jet_pt[i] < 15) continue;
        if ((nanoT.Jet_chEmEF[i] + nanoT.Jet_neEmEF[i]) > 0.9) continue;

        // find minimum ΔR to any muon
        TLorentzVector p4Jet;
        p4Jet.SetPtEtaPhiM(
            nanoT.Jet_pt[i],
            nanoT.Jet_eta[i],
            nanoT.Jet_phi[i],
            nanoT.Jet_mass[i]
        );
        double minDr = std::numeric_limits<double>::infinity();
        for (UInt_t iMu = 0; iMu < nanoT.nMuon; ++iMu) {
            TLorentzVector p4Mu;
            p4Mu.SetPtEtaPhiM(
                nanoT.Muon_pt[iMu],
                nanoT.Muon_eta[iMu],
                nanoT.Muon_phi[iMu],
                nanoT.Muon_mass[iMu]  // or a fixed muon mass if you don't have this branch
            );
            double dr = p4Jet.DeltaR(p4Mu);
            if (dr < minDr) minDr = dr;
            if (minDr < 0.2) break; // skip other muons since we found an overlapping muon
        }
        if(minDr < 0.2) continue; // skip that jet since it overlaps with a muon
   
        // Now check if the jet is in the veto region
        auto jvmNumber = jvmRef->evaluate({jvmKeyName, nanoT.Jet_eta[i], nanoT.Jet_phi[i]});
        // the jvmNumber will be either zero (0.0) or non-zero (100.0).
        // Non-zero means the jet is in the veto region
        if (jvmNumber > 0.0) {
            vetoEvent = true;
            break; // no need to loop over remaining jets
        }
    }//nJet loop

    return vetoEvent;
}


// Request applied to BOTH algos in one pass
struct SystTagDetail {
    std::string setName;   // e.g. "systOnMcSet1" (empty => Nominal)
    std::string tagAK4;    // full AK4 tag, e.g. "Summer19UL18_V5_MC_AbsoluteScale_AK4PFchs"
    std::string tagAK8;    // full AK8 tag, e.g. "Summer19UL18_V5_MC_AbsoluteScale_AK8PFPuppi"
    std::string baseTag;   // e.g. "AbsoluteScale", "Regrouped_BBEC1_2018"
    std::string var;       // "Up" / "Down"

    bool isNominal() const { return setName.empty(); }
    std::string systSetName() const { return isNominal() ? "Nominal" : setName; }
    std::string systName() const { return isNominal() ? "Nominal" : (baseTag + "_" + var); }
};

// unchanged
static TDirectory* getOrMkdir(TDirectory* parent, const std::string& name) {
    if (!parent) return nullptr;
    if (auto* d = dynamic_cast<TDirectory*>(parent->Get(name.c_str()))) return d;
    return parent->mkdir(name.c_str());
}

static std::string baseSystTag(std::string s) {
    // drop leading "..._MC_"
    if (auto p = s.find("_MC_"); p != std::string::npos) s = s.substr(p + 4);
    // drop trailing "_AK..." (algo & particle-flow flavor)
    if (auto p = s.rfind("_AK"); p != std::string::npos) s = s.substr(0, p);
    return s; // e.g. "AbsoluteScale", "Regrouped_BBEC1_2018"
}

// s4/s8: map<setName, vector<fullTag>>
static std::vector<SystTagDetail> buildSystTagDetails(const SystSetMap& s4,
                                                            const SystSetMap& s8)
{
    std::vector<SystTagDetail> systTagDetails;

    for (const auto& [set, tags4] : s4) {
        auto it8 = s8.find(set);
        if (it8 == s8.end()) continue;  // set must exist for both algos

        // map baseTag -> fullTag for each algo
        std::unordered_map<std::string, std::string> map4, map8;
        for (const auto& t : tags4)            map4.emplace(baseSystTag(t), t);
        for (const auto& t : it8->second)      map8.emplace(baseSystTag(t), t);

        // intersect on baseTag
        for (const auto& [base, full4] : map4) {
            auto itFull8 = map8.find(base);
            if (itFull8 == map8.end()) continue;

            for (const char* var : {"Up","Down"}) {
                systTagDetails.push_back(SystTagDetail{
                    /*setName*/ set,
                    /*tagAK4*/  full4,
                    /*tagAK8*/  itFull8->second,
                    /*baseTag*/ base,
                    /*var*/     var
                });
            }
        }
    }
    return systTagDetails;
}


// ===== 3) Hists contain BOTH algos; optional *_Nano only once =====
struct Hists {
    TH1F* hJetPt_AK4_Nano{};
    TH1F* hJetPt_AK8_Nano{};
    TH1F* hMET_Nano{};

    TH1F* hJetPt_AK4{};
    TH1F* hJetPt_AK8{};
    TH1F* hMET{};
};

static Hists makeHists(TFile& fout,
                       const std::string& year,
                       bool isData,
                       const std::string& systSetName,
                       const std::string& systName,
                       bool alsoNano = false)
{
    auto* yearDir = dynamic_cast<TDirectory*>(fout.Get(year.c_str()));
    if (!yearDir) yearDir = fout.mkdir(year.c_str());

    auto* typeDir = getOrMkdir(yearDir, isData ? "Data" : "MC");
    auto* setDir  = getOrMkdir(typeDir, systSetName);
    auto* passDir = getOrMkdir(setDir, systName);
    passDir->cd();

    Hists h{};
    if (alsoNano) {
        h.hJetPt_AK4_Nano = new TH1F("hJetPt_AK4_Nano", "", 50, 0, 500);
        h.hJetPt_AK8_Nano = new TH1F("hJetPt_AK8_Nano", "", 50, 0, 500);
        h.hMET_Nano       = new TH1F("hMET_Nano",       "", 50, 0, 500);
    }
    h.hJetPt_AK4 = new TH1F("hJetPt_AK4", "", 50, 0, 500);
    h.hJetPt_AK8 = new TH1F("hJetPt_AK8", "", 50, 0, 500);
    h.hMET       = new TH1F("hMET",       "", 50, 0, 500);
    return h;
}

// ===== 4) Single-pass over Events: apply to BOTH AK4 & AK8 =====
static void processEvents(const std::string& inputFile,
                              TFile& fout,
                              const std::string& year,
                              bool isData,
                              const std::optional<std::string>& era,
                              const SystTagDetail& systTagDetail)
{
    // --- AK4 refs
    auto cfgAK4  = loadJsonConfig("JercFileAndTagNamesAK4.json");
    Tags tagsAK4 = getTagNames(cfgAK4, year, isData, era);
    CorrectionRefs refsAK4(tagsAK4);

    // --- AK8 refs
    auto cfgAK8  = loadJsonConfig("JercFileAndTagNamesAK8.json");
    Tags tagsAK8 = getTagNames(cfgAK8, year, isData, era);
    CorrectionRefs refsAK8(tagsAK8);

    // --- JVM (unchanged)
    auto cfgJvm   = loadJsonConfig("JvmFileAndTagNames.json");
    const auto& y = cfgJvm.at(year);
    const auto& jvmFile = getTagName(y, "jvmFilePath");
    const auto& jvmTag  = getTagName(y, "jvmTagName");
    const auto& jvmRef  = correction::CorrectionSet::from_file(jvmFile)->at(jvmTag);
    const auto& jvmKey  = getTagName(y, "jvmKeyName");

    // --- Chain & branches
    TChain chain("Events");
    chain.Add(inputFile.c_str());
    NanoTree nanoT;
    setupNanoBranches(&chain, nanoT, isData);

    const bool writeNano = systTagDetail.isNominal(); // write once in Nominal pass
    Hists H = makeHists(fout, year, isData, systTagDetail.systSetName(), systTagDetail.systName(), writeNano);

    const Long64_t nEntries = chain.GetEntries();
    int countVeto = 0;
    bool print = true;
    for (Long64_t i = 0; i < nEntries; ++i) {
        if (chain.LoadTree(i) < 0) break;
        chain.GetTree()->GetEntry(i);

        if (writeNano && H.hMET_Nano) H.hMET_Nano->Fill(nanoT.MET_pt);
        if(print) std::cout<<"   MET from NanoAOD = "<<nanoT.MET_pt<<'\n';

        // Select indices
        std::vector<UInt_t> indicesAK4; indicesAK4.reserve(nanoT.nJet);
        std::vector<UInt_t> indicesAK8; indicesAK8.reserve(nanoT.nFatJet);

        for (UInt_t j=0; j<nanoT.nJet; ++j) {
            if (nanoT.Jet_pt[j] < 15 || std::abs(nanoT.Jet_eta[j]) > 5.2) continue;
            indicesAK4.push_back(j);
            if (writeNano && H.hJetPt_AK4_Nano) H.hJetPt_AK4_Nano->Fill(nanoT.Jet_pt[j]);
        }
        for (UInt_t j=0; j<nanoT.nFatJet; ++j) {
            if (nanoT.FatJet_pt[j] < 15 || std::abs(nanoT.FatJet_eta[j]) > 5.2) continue;
            indicesAK8.push_back(j);
            if (writeNano && H.hJetPt_AK8_Nano) H.hJetPt_AK8_Nano->Fill(nanoT.FatJet_pt[j]);
        }

        // --- Nominal to BOTH
        applyNominalCorrections<AK4Specs>(nanoT, refsAK4, isData, indicesAK4, print);
        applyNominalCorrections<AK8Specs>(nanoT, refsAK8, isData, indicesAK8, print);
        if(print) std::cout<<"   MET after Nominal JERC = "<<nanoT.MET_pt<<'\n';

        // --- Systematic (MC only) to BOTH
        if (!systTagDetail.isNominal() && !isData) {
            // NOTE: Each algo reads its OWN JSON; same (set, tag, var) name, no union.
            applySystematicShift<AK4Specs>(nanoT, refsAK4, systTagDetail.tagAK4, systTagDetail.var, indicesAK4, print);
            applySystematicShift<AK8Specs>(nanoT, refsAK8, systTagDetail.tagAK8, systTagDetail.var, indicesAK8, print);
            if(print) std::cout<<"   MET after Systematic JERC = "<<nanoT.MET_pt<<'\n';
        }
        print = false;

        // Fill hists for BOTH algos
        for (auto idx : indicesAK4) H.hJetPt_AK4->Fill(nanoT.Jet_pt[idx]);
        for (auto idx : indicesAK8) H.hJetPt_AK8->Fill(nanoT.FatJet_pt[idx]);
        H.hMET->Fill(nanoT.MET_pt);

        // Jet veto map (unchanged policy)
        if (checkIfAnyJetInVetoRegion(jvmRef, jvmKey, nanoT)) {
            countVeto++;
            continue;
        }

        // ... further analysis for this pass ...
    }//Event loop

    std::cout<<"   Number of events vetoed due to JetVetoMap: "<<countVeto<<'\n';

}

// ===== 5) Public API: 1x Nominal (both), then correlated systematics (both) =====
void processEventsWithNominalAndSyst(const std::string& inputFile,
                                     TFile& fout,
                                     const std::string& year,
                                     bool isData,
                                     const std::optional<std::string>& era = std::nullopt)
{
    auto cfgAK4 = loadJsonConfig("JercFileAndTagNamesAK4.json");
    auto cfgAK8 = loadJsonConfig("JercFileAndTagNamesAK8.json");

    // Resolve systematic menus independently (no union), then intersect
    SystSetMap s4 = getSystTagNames(cfgAK4, year);
    SystSetMap s8 = getSystTagNames(cfgAK8, year);

    // Nominal: one pass applies to BOTH algos
    std::cout<<" [Nominal] "<<'\n';
    processEvents(inputFile, fout, year, isData, era, SystTagDetail{/*Nominal*/});


    // Correlated systematics (MC only): run only where BOTH algos define (set, tag)
    if (!isData) {
        auto systTagDetails = buildSystTagDetails(s4, s8);
        for (const auto& systTagDetail : systTagDetails) {
            std::cout<<"\n [Syst]: "<<systTagDetail.systName()<<'\n';
            processEvents(inputFile, fout, year, false, era, systTagDetail);
        }
    }
}



// ---------------------------
// Example batch driver
// ---------------------------
void applyJercAndJvmOnNano() {
    const std::string fInputData = "NanoAod_Data.root";
    const std::string fInputMc   = "NanoAod_MC.root";

    // Prepare output file
    std::string outName = "output.root";
    TFile fout(outName.c_str(), "RECREATE");


    std::vector<std::string> mcYears = {
        "2017", 
        //"2018"
    };
    std::vector<std::pair<std::string, std::string>> dataConfigs = {
        {"2017", "Era2017B"},
        //{"2018", "Era2018A"}
    };

    for (const auto& year : mcYears) {
        std::cout<<"[MC] : "<<year <<'\n';
        processEventsWithNominalAndSyst(fInputMc, fout, year, /*isData=*/false);
    }

    for (const auto& [year, era] : dataConfigs) {
        std::cout<<"\n[Data] : "<<year <<" : "<<era <<'\n';
        processEventsWithNominalAndSyst(fInputData, fout, year, /*isData=*/true, era);
    }

    fout.Write();
    fout.Close();
    std::cout << "Wrote output to " << outName << "\n";
}

