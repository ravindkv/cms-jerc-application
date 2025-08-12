/**
 * @file applyJercAndJvmOnNano.C
 * @brief Example ROOT macro demonstrating how to apply jet energy corrections
 *        (JEC), jet energy resolution smearing (JER) and the jet veto map on
 *        NanoAOD samples.
 *
 * The macro reads a NanoAOD ROOT file, loads the appropriate correction and
 * resolution factors from JSON files via `correctionlib`, and writes a ROOT
 * file containing histograms of the corrected jet and MET quantities.  Both
 * Monte Carlo and data workflows are supported.
 *
 * Typical usage from a CMSSW environment:
 *
 * ```
 * cmsenv
 * root -b -q applyJercAndJvmOnNano.C
 * ```
 *
 * The entry point is the function `applyJercAndJvmOnNano()` defined at the end
 * of this file which configures the years and input files to process.
 */
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
/**
 * POD structure mirroring the subset of NanoAOD branches accessed in this
 * macro.  Arrays are sized to the maximum expected number of objects so that
 * the event can be read directly into C++ types.
 */
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

/**
 * Attach the branches of the given \c TChain to the fields of a \c NanoTree
 * instance.  Only the branches needed for the corrections are connected.  For
 * data samples the generator level branches are omitted.
 *
 * @param chain  Input chain containing the NanoAOD tree.
 * @param nanoT  Structure that will receive the branch addresses.
 * @param isData Set to \c true for data samples to skip MC-only branches.
 */
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

/**
 * Accessor helpers for AK4 jets.  The correction routines are written to be
 * generic over the jet collection and rely on these functions to read and
 * modify the appropriate NanoTree fields.
 */
struct AK4Specs {
  static Float_t  getPt (const NanoTree& nt, UInt_t i) { return nt.Jet_pt[i];  }
  static Float_t  getEta(const NanoTree& nt, UInt_t i) { return nt.Jet_eta[i]; }
  static Float_t  getPhi(const NanoTree& nt, UInt_t i) { return nt.Jet_phi[i]; }
  static Float_t  getRawFactor(const NanoTree& nt, UInt_t i){ return nt.Jet_rawFactor[i]; }
  static Float_t  getArea(const NanoTree& nt, UInt_t i)     { return nt.Jet_area[i]; }
  static TLorentzVector makeTLorentzVector(const NanoTree& nt, UInt_t i){
    TLorentzVector p4; p4.SetPtEtaPhiM(nt.Jet_pt[i], nt.Jet_eta[i], nt.Jet_phi[i], nt.Jet_mass[i]); return p4;
  }
  static void applyCorrection(NanoTree& nt, UInt_t i, double sf){
    nt.Jet_pt[i]   *= sf;
    nt.Jet_mass[i] *= sf;
  }

  // --- Gen matching (AK4)
  static int      getGenIdx (const NanoTree& nt, UInt_t i) { return nt.Jet_genJetIdx[i]; }
  static UInt_t   getNGen    (const NanoTree& nt)          { return nt.nGenJet; }
  static Float_t  getGenPt  (const NanoTree& nt, UInt_t j) { return nt.GenJet_pt[j]; }
  static Float_t  getGenEta (const NanoTree& nt, UInt_t j) { return nt.GenJet_eta[j]; }
  static Float_t  getGenPhi (const NanoTree& nt, UInt_t j) { return nt.GenJet_phi[j]; }
  static bool     isValidGenIdx(const NanoTree& nt, int j){
    return (j > -1) && (static_cast<UInt_t>(j) < getNGen(nt));
  }
};

/**
 * Accessor helpers for AK8 jets used by the templated correction routines.
 */
struct AK8Specs {
  static Float_t  getPt (const NanoTree& nt, UInt_t i) { return nt.FatJet_pt[i];  }
  static Float_t  getEta(const NanoTree& nt, UInt_t i) { return nt.FatJet_eta[i]; }
  static Float_t  getPhi(const NanoTree& nt, UInt_t i) { return nt.FatJet_phi[i]; }
  static Float_t  getRawFactor(const NanoTree& nt, UInt_t i){ return nt.FatJet_rawFactor[i]; }
  static Float_t  getArea(const NanoTree& nt, UInt_t i)     { return nt.FatJet_area[i]; }
  static TLorentzVector makeTLorentzVector(const NanoTree& nt, UInt_t i){
    TLorentzVector p4; p4.SetPtEtaPhiM(nt.FatJet_pt[i], nt.FatJet_eta[i], nt.FatJet_phi[i], nt.FatJet_mass[i]); return p4;
  }
  static void applyCorrection(NanoTree& nt, UInt_t i, double sf){
    nt.FatJet_pt[i]   *= sf;
    nt.FatJet_mass[i] *= sf;
  }

  // --- Gen matching (AK8)
  static int      getGenIdx (const NanoTree& nt, UInt_t i) { return nt.FatJet_genJetAK8Idx[i]; }
  static UInt_t   getNGen    (const NanoTree& nt)          { return nt.nGenJetAK8; }
  static Float_t  getGenPt  (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_pt[j]; }
  static Float_t  getGenEta (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_eta[j]; }
  static Float_t  getGenPhi (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_phi[j]; }
  static bool     isValidGenIdx(const NanoTree& nt, int j){
    return (j > -1) && (static_cast<UInt_t>(j) < getNGen(nt));
  }
};



// ---------------------------
// JSON / Tag helpers
// ---------------------------
/**
 * Container for the various tag names needed to look up JEC and JER
 * corrections from the JSON files.  The exact tags depend on year, data/MC
 * state and era.
 */
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

/**
 * Read a JSON configuration file from disk.
 */
nlohmann::json loadJsonConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open JSON config: " + filename);
    }
    nlohmann::json j;
    f >> j;
    return j;
}

/**
 * Helper to extract a string field from a JSON object, throwing a
 * \c runtime_error if the key is missing.
 */
std::string getTagName(const nlohmann::json& obj, const std::string& key) {
    if (!obj.contains(key)) {
        throw std::runtime_error("Missing required key in JSON: " + key);
    }
    return obj.at(key).get<std::string>();
}

/**
 * Gather all relevant tag names for a given year and data-taking scenario.
 * For data the specific era must be provided to select the proper tags.
 */
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

/**
 * Retrieve a correction from a \c CorrectionSet and turn any exception into a
 * descriptive \c runtime_error.
 */
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
/**
 * Convenience wrapper that caches the correction objects needed for a given
 * year/era.  The underlying \c CorrectionSet is cached across calls to avoid
 * repeatedly opening the same JSON file.
 */
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

/**
 * Compute the distance \f$\Delta R\f$ between two objects given their \c eta and
 * \c phi coordinates.
 */
double deltaR(float eta1, float phi1, float eta2, float phi2) {
    double dEta = double(eta1) - double(eta2);
    double dPhi = TVector2::Phi_mpi_pi(phi1 - phi2);
    return std::abs(std::sqrt(dEta*dEta + dPhi*dPhi));
}


// 2) A single templated nominal corrections, generic over Specs (AK4Specs or AK8Specs) ----
/**
 * Apply the nominal jet energy corrections (undo raw factors, L1, L2 and
 * optionally the residual step for data) to a set of jet indices.
 *
 * @tparam Specs  Helper struct describing how to access a particular jet
 *                collection (e.g. AK4 or AK8).
 * @param nanoT   Event record to modify.
 * @param refs    Pre-loaded correction references.
 * @param isData  Whether the event comes from data (controls residual step).
 * @param idxs    Indices of jets to correct.
 * @param print   When true, print debug information for the first event.
 */
template<typename Specs>
void applyNominalCorrections(NanoTree& nanoT,
                       CorrectionRefs& refs,
                       bool isData,
                       const std::vector<UInt_t>& idxs,
                       bool print=false)
{
    for(auto idx: idxs){
        if(print) std::cout<<"    [Jet] index="<<idx<<"\n";
        nanoT.MET_pt += Specs::getPt(nanoT,idx);//add MET to jet 
        if(print) std::cout<<"      default NanoAod Pt="<<Specs::getPt(nanoT,idx)<<"\n";

        // raw correction
        double rawSF = 1.0 - Specs::getRawFactor(nanoT, idx);
        Specs::applyCorrection(nanoT, idx, rawSF);
        if(print) std::cout<<"      after undoing   Pt="<<Specs::getPt(nanoT,idx)<<"\n";

        // L1
        double c1 = refs.corrL1->evaluate({ Specs::getArea(nanoT,idx),
                                            Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx),
                                            nanoT.Rho });
        Specs::applyCorrection(nanoT, idx, c1);
        if(print) std::cout<<"      after L1    Pt="<<Specs::getPt(nanoT,idx)<<"\n";

        // L2 rel
        double c2 = refs.corrL2->evaluate({ Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx) });
        Specs::applyCorrection(nanoT, idx, c2);
        if(print) std::cout<<"      after L2Rel Pt="<<Specs::getPt(nanoT,idx)<<"\n";

        // Residual (data only)
        if(isData){
          double cR = refs.corrL2ResL3Res->evaluate({ Specs::getEta(nanoT,idx),
                                                      Specs::getPt(nanoT,idx) });
          Specs::applyCorrection(nanoT, idx, cR);
          if(print) std::cout<<"     after ResL3 Pt="<<Specs::getPt(nanoT,idx)<<"\n";
        }
        nanoT.MET_pt -= Specs::getPt(nanoT,idx);//substract MET from jet 
    }
}


// ----- JER uncertainty bins from JSON -----
// One bin entry
/**
 * Definition of a single JER uncertainty bin as provided in the JSON config.
 * The bin is identified by a label and optionally restricted to a region in
 * \f$\eta\f$ and \f$p_T\f$.
 */
struct JerBin {
    std::string label;   // e.g. "CMS_res_j_2017_absEta0to1p93_pT0toInf" or "CMS_res_j_2017"
    double etaMin{}, etaMax{}, ptMin{}, ptMax{};
};

// Map<setName, vector<JerBin>>, e.g. "ShiftedJERFull" -> [bins], "ShiftedJERTotal" -> [1 bin]
using JerSetMap = std::map<std::string, std::vector<JerBin>>;

/**
 * Parse the JER uncertainty definitions from the JSON configuration for a
 * given year.
 */
static JerSetMap getJerUncertaintySets(const nlohmann::json& baseJson, const std::string& year) {
    JerSetMap out;
    if (!baseJson.contains(year)) return out;
    const auto& y = baseJson.at(year);
    if (!y.contains("ShiftedJER")) return out;

    const auto& j = y.at("ShiftedJER");
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string setName = it.key();          // "ShiftedJERFull" or "ShiftedJERTotal"
        const auto& obj = it.value();                  // object of { label -> {etaMin,...} }

        if (!obj.is_object()) continue;
        std::vector<JerBin> bins;
        bins.reserve(obj.size());
        for (auto it2 = obj.begin(); it2 != obj.end(); ++it2) {
            const std::string label = it2.key();
            const auto& def = it2.value();
            JerBin b;
            b.label = label;
            b.etaMin = def.value("etaMin", 0.0);
            b.etaMax = def.value("etaMax", 999.0);
            b.ptMin  = def.value("ptMin",  0.0);
            b.ptMax  = def.value("ptMax",  1e9);
            bins.push_back(b);
        }
        out.emplace(setName, std::move(bins));
    }
    return out;
}


//--------------------------------------------------
// Systematic Uncertanities
//--------------------------------------------------
// unchanged
static TDirectory* getOrMkdir(TDirectory* parent, const std::string& name) {
    if (!parent) return nullptr;
    if (auto* d = dynamic_cast<TDirectory*>(parent->Get(name.c_str()))) return d;
    return parent->mkdir(name.c_str());
}

// Each entry: { fullTag, base/custom name }
using SystPair   = std::pair<std::string, std::string>;
using SystSetMap = std::map<std::string, std::vector<SystPair>>;

/**
 * Extract the JES systematic tag names for a given year from the JSON
 * configuration.
 */
SystSetMap getSystTagNames(const nlohmann::json& baseJson, const std::string& year) {
    SystSetMap out;
    if (!baseJson.contains(year)) return out;

    const auto& yearObj = baseJson.at(year);
    if (!yearObj.contains("ShiftedJES")) return out;

    const auto& systs = yearObj.at("ShiftedJES");
    for (auto it = systs.begin(); it != systs.end(); ++it) {
        const std::string setName = it.key();
        const auto& arr = it.value();
        if (!arr.is_array()) continue;

        std::vector<SystPair> pairs;
        pairs.reserve(arr.size());

        for (const auto& item : arr) {
            if (item.is_array() && item.size() >= 2 && item.at(0).is_string() && item.at(1).is_string()) {
                pairs.emplace_back(item.at(0).get<std::string>(), item.at(1).get<std::string>());
            } 
        }

        out.emplace(setName, std::move(pairs));
    }
    return out;
}

// add a kind flag + optional region for JER
enum class SystKind { Nominal, JES, JER };

/**
 * Description of a single systematic variation to be applied.  For JES
 * systematics it stores the tags for AK4 and AK8 corrections, while for JER
 * systematics it records the affected `JerBin` region.
 */
struct SystTagDetail {
    // JES fields (unchanged)
    std::string setName;   // e.g. "ShiftedJESFull" or "ShiftedJERFull" (for JER we reuse this)
    std::string tagAK4;    // JES only
    std::string tagAK8;    // JES only
    std::string baseTag;   // JES: CustomName; JER: label (e.g. "CMS_res_j_2017_absEta…")
    std::string var;       // "Up" / "Down" (empty for nominal)
    // new:
    SystKind kind{SystKind::Nominal};
    std::optional<JerBin> jerRegion;  // only set for JER

    bool isNominal() const { return kind == SystKind::Nominal; }
    std::string systSetName() const { return isNominal() ? "Nominal" : setName; }
    std::string systName() const {
        if (isNominal()) return "Nominal";
        if (kind == SystKind::JES) return baseTag + "_" + var;
        // JER: use label_up/down
        return baseTag + "_" + var;
    }
};


// s4/s8: map<setName, vector<{fullTag, base/custom}>>
/**
 * Combine the JES systematic tag information for AK4 and AK8 jets.  Only
 * systematics defined for both algorithms are returned.
 */
static std::vector<SystTagDetail> buildSystTagDetails(const SystSetMap& s4,
                                                      const SystSetMap& s8)
{
    std::vector<SystTagDetail> systTagDetails;

    for (const auto& [set, pairs4] : s4) {
        auto it8 = s8.find(set);
        if (it8 == s8.end()) continue;  // set must exist for both algos
        const auto& pairs8 = it8->second;

        // Map base/custom name -> fullTag for each algo
        std::unordered_map<std::string, std::string> map4, map8;
        map4.reserve(pairs4.size());
        map8.reserve(pairs8.size());

        for (const auto& p : pairs4) map4.emplace(p.second, p.first); // key = base/custom
        for (const auto& p : pairs8) map8.emplace(p.second, p.first);

        // Intersect on base/custom name (2nd element)
        for (const auto& [base, full4] : map4) {
            auto itFull8 = map8.find(base);
            if (itFull8 == map8.end()) continue;

            for (const char* var : {"Up","Down"}) {
                systTagDetails.push_back(SystTagDetail{
                    /*setName*/ set,
                    /*tagAK4*/  full4,
                    /*tagAK8*/  itFull8->second,
                    /*baseTag*/ base,   // <-- this is the CustomName
                    /*var*/     var,
                    /*kind*/    SystKind::JES,
                    /*jerRegion*/ std::nullopt
                });
            }
        }
    }
    return systTagDetails;
}

// Build JER details: each (setName, JerBin) × {Up,Down}
/**
 * Expand the JER uncertainty definition into explicit up/down variations for
 * each bin defined in the JSON configuration.
 */
static std::vector<SystTagDetail> buildJerTagDetails(const JerSetMap& jerSets) {
    std::vector<SystTagDetail> out;
    for (const auto& [setName, bins] : jerSets) {
        for (const auto& b : bins) {
            for (const char* var : {"Up","Down"}) {
                SystTagDetail d;
                d.setName   = setName;        // "ShiftedJERFull" or "ShiftedJERTotal"
                d.tagAK4    = "";             // not used for JER
                d.tagAK8    = "";             // not used for JER
                d.baseTag   = b.label;        // the label/key from JSON
                d.var       = var;
                d.kind      = SystKind::JER;
                d.jerRegion = b;
                out.push_back(std::move(d));
            }
        }
    }
    return out;
}

/**
 * Apply JER smearing or its variations to a set of jets.  When a region is
 * provided the smearing is only modified for jets inside that bin, while jets
 * outside keep the nominal scale factors.
 */
template<typename Specs>
void applyJEROnly(NanoTree& nanoT,
                  CorrectionRefs& refs,
                  const std::vector<UInt_t>& idxs,
                  const std::string& var,                          // "nom" / "up" / "down"
                  const std::optional<JerBin>& region = std::nullopt,
                  bool print=false)
{
    if (idxs.empty()) return;
    // Data: no JER
    if (std::is_same<Specs, AK4Specs>::value || std::is_same<Specs, AK8Specs>::value) {
        // ok
    }
    // guard: MC only
    // We assume caller ensures MC; if you prefer, pass isData and early return.

    for (auto idx : idxs) {
        const double etaJet = Specs::getEta(nanoT, idx);
        const double ptJet  = Specs::getPt (nanoT, idx);
        const double phiJet = Specs::getPhi(nanoT, idx);

        // region gating: use |eta|
        auto inRegion = [&](const JerBin& b){
            const double aeta = std::fabs(etaJet);
            return (aeta >= b.etaMin && aeta < b.etaMax &&
                    ptJet >= b.ptMin   && ptJet < b.ptMax);
        };

        // choose which var to use for THIS jet
        std::string useVar = "nom";
        if (region.has_value()) {
            useVar = inRegion(*region) ? var : "nom";
        } else {
            useVar = var;
        }

        nanoT.MET_pt += Specs::getPt(nanoT, idx);

        const double reso = refs.corrJerReso->evaluate({ etaJet, ptJet, nanoT.Rho });
        const double sf   = refs.corrJerSf->evaluate({ etaJet, useVar });

        refs.randomGen.SetSeed(nanoT.event + nanoT.run + nanoT.luminosityBlock);

        const int genIdx = Specs::getGenIdx(nanoT, idx);
        bool isMatch = false;
        if (Specs::isValidGenIdx(nanoT, genIdx)) {
            const double dR = deltaR(phiJet, Specs::getGenPhi(nanoT, genIdx),
                                     etaJet, Specs::getGenEta(nanoT, genIdx));
            if (dR < 0.2 &&
                std::abs(ptJet - Specs::getGenPt(nanoT, genIdx)) < 3.0 * reso * ptJet) {
                isMatch = true;
            }
        }

        double corr = 1.0;
        if (isMatch) {
            corr = std::max(0.0, 1.0 + (sf - 1.0) * (ptJet - Specs::getGenPt(nanoT, genIdx)) / ptJet);
        } else {
            corr = std::max(0.0, 1.0 + refs.randomGen.Gaus(0.0, reso) *
                                     std::sqrt(std::max(sf*sf - 1.0, 0.0)));
        }
        Specs::applyCorrection(nanoT, idx, corr);

        if (print) {
            std::cout<<"     JER("<<useVar<<")  Pt="<<Specs::getPt(nanoT, idx);
            if (region.has_value()) {
                std::cout<<"   [inRegion="<<(inRegion(*region)?"yes":"no")<<"]";
            }
            std::cout<<"\n";
        }

        nanoT.MET_pt -= Specs::getPt(nanoT, idx);
    }
}


// 4) Templated systematic shifts
/**
 * Apply a JES systematic variation.  The specific correction is selected by
 * \p systName and scaled up or down according to \p var.
 */
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
        nanoT.MET_pt += Specs::getPt(nanoT,idx);//add MET to jet 

        if(print) std::cout<<"    [Jet] index="<<idx<<"\n";
        if(print) std::cout<<"     Nominal corrected    Pt="<<Specs::getPt(nanoT,idx)<<"\n";
        double scale = systCorr->evaluate({ Specs::getEta(nanoT,idx), Specs::getPt(nanoT,idx) });
        double shift = (var=="Up" ? 1+scale : 1-scale);
        Specs::applyCorrection(nanoT, idx, shift);
        if(print) std::cout<<"     Syst corrected    Pt="<<Specs::getPt(nanoT,idx)<<"\n";

        nanoT.MET_pt -= Specs::getPt(nanoT,idx);//substract MET from jet 
    }
}

//--------------------------------------------------
// Jet Veto Map
//--------------------------------------------------
/**
 * Check whether any reconstructed jet falls inside a jet veto region as
 * defined by the provided correction map.
 */
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
        double minDr = std::numeric_limits<double>::infinity();
        for (UInt_t iMu = 0; iMu < nanoT.nMuon; ++iMu) {
            double dr = deltaR(nanoT.Jet_eta[i], nanoT.Jet_phi[i], 
                                                  nanoT.Muon_eta[iMu], nanoT.Muon_phi[iMu]);
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

//--------------------------------------------------
// Store Histograms for Sanity Checks
//--------------------------------------------------
/**
 * Small collection of histograms used to monitor the effect of the
 * corrections.  When running the nominal pass additional histograms containing
 * the uncorrected NanoAOD values can also be filled.
 */
struct Hists {
    TH1F* hJetPt_AK4_Nano{};
    TH1F* hJetPt_AK8_Nano{};
    TH1F* hMET_Nano{};

    TH1F* hJetPt_AK4{};
    TH1F* hJetPt_AK8{};
    TH1F* hMET{};
};

/**
 * Create and organise the monitoring histograms in the output file.  The
 * directory structure is `year/type/systematicSet/systematicName`.
 */
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
        h.hJetPt_AK4_Nano = new TH1F("hJetPt_AK4_Nano", "", 50, 10, 510);
        h.hJetPt_AK8_Nano = new TH1F("hJetPt_AK8_Nano", "", 50, 10, 510);
        h.hMET_Nano       = new TH1F("hMET_Nano",       "", 50, 10, 510);
    }
    h.hJetPt_AK4 = new TH1F("hJetPt_AK4", "", 50, 10, 510);
    h.hJetPt_AK8 = new TH1F("hJetPt_AK8", "", 50, 10, 510);
    h.hMET       = new TH1F("hMET",       "", 50, 10, 510);
    return h;
}

//--------------------------------------------------
// Events are looped in this function
//--------------------------------------------------
/**
 * Core event loop applying the nominal corrections and, depending on
 * \c systTagDetail, an additional JES or JER systematic variation.  Histograms
 * are filled and written to the provided output file.
 */
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
    int printCount = 0;
    bool print = false;
    for (Long64_t i = 0; i < nEntries; ++i) {
        if (chain.LoadTree(i) < 0) break;
        chain.GetTree()->GetEntry(i);

        if (writeNano && H.hMET_Nano) H.hMET_Nano->Fill(nanoT.MET_pt);

        // --- select jet indices (unchanged, your overlap logic)
        std::vector<UInt_t> indicesAK4; indicesAK4.reserve(nanoT.nJet);
        std::vector<UInt_t> indicesAK8; indicesAK8.reserve(nanoT.nFatJet);

        // AK8
        for (UInt_t j = 0; j < nanoT.nFatJet; ++j) {
            if (nanoT.FatJet_pt[j] < 100 || std::abs(nanoT.FatJet_eta[j]) > 5.2) continue;
            indicesAK8.push_back(j);
            if (writeNano && H.hJetPt_AK8_Nano) H.hJetPt_AK8_Nano->Fill(nanoT.FatJet_pt[j]);
        }
        // AK4 non-overlapping
        for (UInt_t j = 0; j < nanoT.nJet; ++j) {
            if (nanoT.Jet_pt[j] < 15 || std::abs(nanoT.Jet_eta[j]) > 5.2) continue;
            bool overlaps = false;
            for (auto k : indicesAK8) {
                if (deltaR(nanoT.Jet_eta[j], nanoT.Jet_phi[j],
                           nanoT.FatJet_eta[k], nanoT.FatJet_phi[k]) < 0.6) {
                    overlaps = true; break;
                }
            }
            if (overlaps) continue;
            indicesAK4.push_back(j);
            if (writeNano && H.hJetPt_AK4_Nano) H.hJetPt_AK4_Nano->Fill(nanoT.Jet_pt[j]);
        }

        if(!indicesAK4.empty() && !indicesAK8.empty() && printCount==0){ printCount++; print = true; }

        if(print) std::cout<<"   MET From NanoAOD = "<<nanoT.MET_pt<<'\n';

        // =========================
        // 1) JES (nominal only here) — NO JER inside
        // =========================
        if(print) std::cout<<"   AK4 (JES nominal)\n";
        applyNominalCorrections<AK4Specs>(nanoT, refsAK4, isData, indicesAK4, print);

        if(print) std::cout<<"   AK8 (JES nominal)\n";
        applyNominalCorrections<AK8Specs>(nanoT, refsAK8, isData, indicesAK8, print);

        if(print) std::cout<<"   MET After (JES nominal) = "<<nanoT.MET_pt<<'\n';

        // =========================
        // 2) JES Uncertainty (MC only), if this pass is JES
        // =========================
        if (!isData && systTagDetail.kind == SystKind::JES) {
            if(print) std::cout<<"   AK4 (JES "<<systTagDetail.var<<")\n";
            applySystematicShift<AK4Specs>(nanoT, refsAK4, systTagDetail.tagAK4, systTagDetail.var, indicesAK4, print);

            if(print) std::cout<<"   AK8 (JES "<<systTagDetail.var<<")\n";
            applySystematicShift<AK8Specs>(nanoT, refsAK8, systTagDetail.tagAK8, systTagDetail.var, indicesAK8, print);

            if(print) std::cout<<"   MET After (JES "<<systTagDetail.var<<") = "<<nanoT.MET_pt<<'\n';
        }

        // =========================
        // 3) JER (after all JES): Nominal or Up/Down in region (MC only)
        // =========================
        if (!isData) {
            if (systTagDetail.kind == SystKind::JER) {
                // up/down only in the specified region; outside region use "nom"
                if(print) std::cout<<"   AK4 (JER "<<systTagDetail.var<<")\n";
                applyJEROnly<AK4Specs>(nanoT, refsAK4, indicesAK4, 
                                       std::string(systTagDetail.var == "Up" ? "up":"down"),
                                       systTagDetail.jerRegion, print);

                if(print) std::cout<<"   AK8 (JER "<<systTagDetail.var<<")\n";
                applyJEROnly<AK8Specs>(nanoT, refsAK8, indicesAK8, 
                                       std::string(systTagDetail.var == "Up" ? "up":"down"),
                                       systTagDetail.jerRegion, print);
                if(print) std::cout<<"   MET After (JER "<<systTagDetail.var<<") = "<<nanoT.MET_pt<<'\n';
            } else {
                // Nominal or JES pass → apply JER(nom) to all jets
                if(print) std::cout<<"   AK4 (JER nom)\n";
                applyJEROnly<AK4Specs>(nanoT, refsAK4, indicesAK4, "nom", std::nullopt, print);
                if(print) std::cout<<"   AK8 (JER nom)\n";
                applyJEROnly<AK8Specs>(nanoT, refsAK8, indicesAK8, "nom", std::nullopt, print);
                if(print) std::cout<<"   MET After (JER nominal) = "<<nanoT.MET_pt<<'\n';
            }
        }

        if(print) std::cout<<"   MET after JERC = "<<nanoT.MET_pt<<'\n';
        print = false;

        // Fill hists
        for (auto idx : indicesAK4) H.hJetPt_AK4->Fill(nanoT.Jet_pt[idx]);
        for (auto idx : indicesAK8) H.hJetPt_AK8->Fill(nanoT.FatJet_pt[idx]);
        H.hMET->Fill(nanoT.MET_pt);

        // Jet veto map
        if (checkIfAnyJetInVetoRegion(jvmRef, jvmKey, nanoT)) {
            countVeto++;
            continue;
        }

        // ... further analysis ...
    }

    std::cout<<"   Number of events vetoed due to JetVetoMap: "<<countVeto<<'\n';
}


//--------------------------------------------------
// Event looping function is called for Nominal and Systematics
//--------------------------------------------------
/**
 * Helper that runs the core event loop for the nominal pass and for all
 * requested JES and JER systematic variations for a given year.  Systematic
 * definitions are retrieved from the JSON configuration files.
 */
void processEventsWithNominalAndSyst(const std::string& inputFile,
                                     TFile& fout,
                                     const std::string& year,
                                     bool isData,
                                     const std::optional<std::string>& era = std::nullopt)
{
    auto cfgAK4 = loadJsonConfig("JercFileAndTagNamesAK4.json");
    auto cfgAK8 = loadJsonConfig("JercFileAndTagNamesAK8.json");

    // JES sets (unchanged)
    SystSetMap s4 = getSystTagNames(cfgAK4, year);
    SystSetMap s8 = getSystTagNames(cfgAK8, year);

    // JER sets (read once; we can use AK4 file as the source of binning)
    JerSetMap jerSets = getJerUncertaintySets(cfgAK4, year);

    // 0) Nominal
    std::cout<<" [Nominal]\n";
    processEvents(inputFile, fout, year, isData, era, SystTagDetail{/*Nominal*/});

    if (!isData) {
        // 1) Correlated JES systematics (only where both algos define the same custom base)
        auto jesDetails = buildSystTagDetails(s4, s8);
        for (const auto& d : jesDetails) {
            std::cout<<"\n [JES Syst]: "<<d.systName()<<'\n';
            processEvents(inputFile, fout, year, false, era, d);
        }

        // 2) JER systematics (Up/Down) with region gating from JSON
        auto jerDetails = buildJerTagDetails(jerSets);
        for (const auto& d : jerDetails) {
            std::cout<<"\n [JER Syst]: "<<d.systSetName()<<"/"<<d.systName()<<'\n';
            processEvents(inputFile, fout, year, false, era, d);
        }
    }
}



//--------------------------------------------------
// Run for multiple years, MC, Data
//--------------------------------------------------
/**
 * Macro entry point.  Configures the input files and years to process and
 * orchestrates running over MC and data samples.
 */
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
        //processEventsWithNominalAndSyst(fInputData, fout, year, /*isData=*/true, era);
    }

    fout.Write();
    fout.Close();
    std::cout << "Wrote output to " << outName << "\n";
}

