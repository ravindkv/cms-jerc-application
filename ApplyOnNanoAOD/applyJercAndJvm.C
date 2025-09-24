/**
 * @file applyJercAndJvm.C
 * @brief Example ROOT macro demonstrating how to apply jet energy corrections
 *        (JEC = JES correction + JER correction) and the jet veto map on
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
 * root -b -q applyJercAndJvm.C
 * ```
 *
 * The entry point is the function `applyJercAndJvm()` defined at the end
 * of this file which configures the years and input files to process.
 */
// Implementation details for applyJercAndJvm.C

#if defined(__CLING__)
#pragma cling add_include_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/include")
#pragma cling add_library_path("$HOME/.local/lib/python3.9/site-packages/correctionlib/lib")
#pragma cling load("libcorrectionlib.so")
#endif

#include <TChain.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TVector2.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>


#include <nlohmann/json.hpp>
#include <correction.h>
#include <boost/timer/progress_display.hpp>


// Global flags to control which jet collections receive corrections.
// Set exactly one of these flags to true to choose the target collection mix.
bool applyOnlyOnAK4 = false;
bool applyOnlyOnAK8 = true;
bool applyOnAK4AndAK8 = false;
// Control whether jet energy corrections are propagated to MET.
bool applyOnMET = true;

// ---------------------------
// NanoAOD flat branches
// ---------------------------
/**
 * Subset of NanoAOD branches accessed in this
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
    Float_t   RawMET_pt{};
    Float_t   RawMET_phi{};
    Float_t   RawPuppiMET_phi{};
    Float_t   RawPuppiMET_pt{};
    Int_t     nJet{};
    Float_t   Jet_pt[200]{};
    Float_t   Jet_eta[200]{};
    Float_t   Jet_phi[200]{};
    Float_t   Jet_mass[200]{};
    Float_t   Jet_rawFactor[200]{};
    Float_t   Jet_muonSubtrFactor[200]{};
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
    chain->SetBranchAddress("RawMET_pt", &nanoT.RawMET_pt);
    chain->SetBranchAddress("RawMET_phi", &nanoT.RawMET_phi);
    chain->SetBranchAddress("RawPuppiMET_pt", &nanoT.RawPuppiMET_pt);
    chain->SetBranchAddress("RawPuppiMET_phi", &nanoT.RawPuppiMET_phi);
    chain->SetBranchAddress("nJet", &nanoT.nJet);
    chain->SetBranchAddress("Jet_pt", &nanoT.Jet_pt);
    chain->SetBranchAddress("Jet_eta", &nanoT.Jet_eta);
    chain->SetBranchAddress("Jet_phi", &nanoT.Jet_phi);
    chain->SetBranchAddress("Jet_mass", &nanoT.Jet_mass);
    chain->SetBranchAddress("Jet_rawFactor", &nanoT.Jet_rawFactor);
	chain->SetBranchAddress("Jet_muonSubtrFactor"  , &nanoT.Jet_muonSubtrFactor);
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
    if(!isData){// Only for MC samples; these branches are required for JER smearing.
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

/**
 * Accessor helpers for AK4 jets specs (specifications).  
 * The correction routines are written to be
 * generic over the jet collection and rely on these functions to read and
 * modify the appropriate NanoTree fields.
 */
struct AK4Specs {
  static Float_t  getPt (const NanoTree& nt, UInt_t i) { return nt.Jet_pt[i];  }
  static Float_t  getEta(const NanoTree& nt, UInt_t i) { return nt.Jet_eta[i]; }
  static Float_t  getPhi(const NanoTree& nt, UInt_t i) { return nt.Jet_phi[i]; }
  static Float_t  getRawFactor(const NanoTree& nt, UInt_t i){ return nt.Jet_rawFactor[i]; }
  static Float_t  getArea(const NanoTree& nt, UInt_t i)     { return nt.Jet_area[i]; }
  static Float_t  getChmEf(const NanoTree& nt, UInt_t i)     { return nt.Jet_chEmEF[i]; }
  static Float_t  getNeEmEF(const NanoTree& nt, UInt_t i)     { return nt.Jet_neEmEF[i]; }
  static Float_t  getMuonSubtrFactor(const NanoTree& nt, UInt_t i)     { return nt.Jet_muonSubtrFactor[i]; }
  static TLorentzVector makeTLorentzVector(const NanoTree& nt, UInt_t i){
    TLorentzVector p4; 
    p4.SetPtEtaPhiM(nt.Jet_pt[i], nt.Jet_eta[i], nt.Jet_phi[i], nt.Jet_mass[i]); 
    return p4;
  }
  static void applyCorrection(NanoTree& nt, UInt_t i, double sf){
    nt.Jet_pt[i]   *= sf;
    nt.Jet_mass[i] *= sf;
  }
  // --- For gen matching (AK4)
  static int      getGenIdx (const NanoTree& nt, UInt_t i) { return nt.Jet_genJetIdx[i]; }
  static UInt_t   getNGen   (const NanoTree& nt)           { return nt.nGenJet; }
  static Float_t  getGenPt  (const NanoTree& nt, UInt_t j) { return nt.GenJet_pt[j]; }
  static Float_t  getGenEta (const NanoTree& nt, UInt_t j) { return nt.GenJet_eta[j]; }
  static Float_t  getGenPhi (const NanoTree& nt, UInt_t j) { return nt.GenJet_phi[j]; }
  static bool     isValidGenIdx(const NanoTree& nt, int j){
    return (j > -1) && (static_cast<UInt_t>(j) < getNGen(nt));
  }
  constexpr static double MINDR = 0.2;
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
    TLorentzVector p4; 
    p4.SetPtEtaPhiM(nt.FatJet_pt[i], nt.FatJet_eta[i], nt.FatJet_phi[i], nt.FatJet_mass[i]); 
    return p4;
  }
  static void applyCorrection(NanoTree& nt, UInt_t i, double sf){
    nt.FatJet_pt[i]   *= sf;
    nt.FatJet_mass[i] *= sf;
  }
  // --- For gen matching (AK8)
  static int      getGenIdx (const NanoTree& nt, UInt_t i) { return nt.FatJet_genJetAK8Idx[i]; }
  static UInt_t   getNGen    (const NanoTree& nt)          { return nt.nGenJetAK8; }
  static Float_t  getGenPt  (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_pt[j]; }
  static Float_t  getGenEta (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_eta[j]; }
  static Float_t  getGenPhi (const NanoTree& nt, UInt_t j) { return nt.GenJetAK8_phi[j]; }
  static bool     isValidGenIdx(const NanoTree& nt, int j){
    return (j > -1) && (static_cast<UInt_t>(j) < getNGen(nt));
  }
  constexpr static double MINDR = 0.4;
};

// ---------------------------
// JSON / Tag helpers
// ---------------------------
/**
 * Container for the various tag names needed to look up JES and JER
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

/**
 * Convenience wrapper that caches the correction objects needed for a given
 * year/era.  The underlying \c CorrectionSet is cached across calls to avoid
 * repeatedly opening the same JSON file.
 */
struct CorrectionRefs {
    std::shared_ptr<correction::CorrectionSet> cs;
    correction::Correction::Ref corrRefJesL1FastJet;
    correction::Correction::Ref corrRefJesL2Relative;
    correction::Correction::Ref corrRefJesL2ResL3Res;
    correction::Correction::Ref corrRefJerReso;
    correction::Correction::Ref corrRefJerSf;
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

        corrRefJesL1FastJet         = safeGet(cs, tags.tagNameL1FastJet);
        corrRefJesL2Relative         = safeGet(cs, tags.tagNameL2Relative);
        corrRefJesL2ResL3Res = safeGet(cs, tags.tagNameL2L3Residual);
        corrRefJerReso    = safeGet(cs, tags.tagNamePtResolution);
        corrRefJerSf      = safeGet(cs, tags.tagNameJerScaleFactor);

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

// ------------------------------------------------------------------
// Helper utilities to centralise common year-based logic
// ------------------------------------------------------------------

/// Years in which the L2Relative correction depends on \f$\phi\f$.
inline bool hasPhiDependentL2(const std::string& year) {
    return year == "2023Post" || year == "2024";
}

/// Years requiring a run-based residual correction (data only).
inline bool requiresRunBasedResidual(const std::string& year) {
    return year == "2023Pre" || year == "2023Post" || year == "2024";
}

/// Years for which the raw MET comes from the PuppiMET branches and
/// the JER scale factors depend on \f$\eta, p_T, \rho\f$.
inline bool usesPuppiMet(const std::string& year) {
    static const std::unordered_set<std::string> years =
        {"2022Pre", "2022Post", "2023Pre", "2023Post", "2024"};
    return years.count(year);
}

/// Select a representative run number for residual corrections when
/// working with very small example input files.
inline double representativeRunNumber(const NanoTree& nanoT, const std::string& year) {
    if (year == "2023Pre")  return 367080.0;
    if (year == "2023Post") return 369803.0;
    if (year == "2024")     return 379412.0;
    return static_cast<double>(nanoT.run);
}

// Simple helper to conditionally print debug information with optional indentation.
namespace {
constexpr int INDENT_BLOCK = 3;
constexpr int INDENT_JET = 4;
constexpr int INDENT_DETAIL = 5;
constexpr int INDENT_STEP = 6;
}
template<typename... Args>
void printDebug(bool enable, int indent, const Args&... args) {
    if (!enable) return;
    for (int i = 0; i < indent; ++i) std::cout.put(' ');
    (std::cout << ... << args) << '\n';
}

//--------------------------------------------------
// Nominal JES Correction
//--------------------------------------------------
// Single templated nominal correction routine shared by both jet specifications.
/**
 * Apply the nominal jet energy corrections (undo raw factors, apply L1, L2 and
 * optionally the residual step for data) to a set of jet indices.
 *
 * @tparam Specs  Helper struct describing how to access a particular jet
 *                collection (e.g. AK4 or AK8).
 * @param nanoT   Event record to modify.
 * @param refs    Pre-loaded correction references.
 * @param isData  Whether the event comes from data (controls residual step).
 * @param idxs    Indices of jets to correct.
 * @param print   When true, print debug information.
 */
template<typename Specs>
void applyJESNominal(NanoTree& nanoT,
                       const std::string& year,
                       CorrectionRefs& refs,
                       bool isData,
                       const std::vector<UInt_t>& idxs,
                       bool print=false)
{
    for(auto idx: idxs){
        printDebug(print, INDENT_JET, "[Jet] index=", idx, 
                                          ", eta= ", Specs::getEta(nanoT,idx),
                                          ", phi= ", Specs::getPhi(nanoT,idx),
                                          ", area= ", Specs::getArea(nanoT,idx),
                                          ", rawFactor= ", Specs::getRawFactor(nanoT,idx)
                                          );
        printDebug(print, INDENT_STEP, "default NanoAod  Pt=", Specs::getPt(nanoT,idx));

        // Raw pT: undo the default JES correction applied in NanoAOD.
        double rawSF = 1.0 - Specs::getRawFactor(nanoT, idx);
        Specs::applyCorrection(nanoT, idx, rawSF);
        printDebug(print, INDENT_STEP, "after undoing    Pt=", Specs::getPt(nanoT,idx));

        // L1FastJet (pileup correction).
        double c1 = refs.corrRefJesL1FastJet->evaluate({ Specs::getArea(nanoT,idx),
                                            Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx),
                                            nanoT.Rho });
        Specs::applyCorrection(nanoT, idx, c1);
        printDebug(print, INDENT_STEP, "after L1FastJet  Pt=", Specs::getPt(nanoT,idx));

        // L2Relative (MC truth correction).
        double c2 = 1.0;
        if(hasPhiDependentL2(year)){
            c2 = refs.corrRefJesL2Relative->evaluate({ Specs::getEta(nanoT,idx),
                                            Specs::getPhi(nanoT,idx),
                                            Specs::getPt(nanoT,idx) });
        }else{
            c2 = refs.corrRefJesL2Relative->evaluate({ Specs::getEta(nanoT,idx),
                                            Specs::getPt(nanoT,idx) });
        }
        Specs::applyCorrection(nanoT, idx, c2);
        printDebug(print, INDENT_STEP, "after L2Relative Pt=", Specs::getPt(nanoT,idx));

        // L2L3Residual (L2Residual + L3Residual together) applied only to data.
        // (Covers residual differences between data and simulation.)
        if(isData){
          double cR = 1.0;
          if(requiresRunBasedResidual(year)){
            const double runNumber = representativeRunNumber(nanoT, year);
            cR = refs.corrRefJesL2ResL3Res->evaluate({runNumber,
                                                      Specs::getEta(nanoT,idx),
                                                      Specs::getPt(nanoT,idx) });
          }else{
            cR = refs.corrRefJesL2ResL3Res->evaluate({ Specs::getEta(nanoT,idx),
                                                      Specs::getPt(nanoT,idx) });
          }
          Specs::applyCorrection(nanoT, idx, cR);
          printDebug(print, INDENT_STEP, "after L2L3Residual Pt=", Specs::getPt(nanoT,idx));
        }
    }
}

//--------------------------------------------------
// Nominal or Up/Down JER Correction 
//--------------------------------------------------
/**
 * Apply JER smearing or its variations to a set of jets.  When a region is
 * provided the smearing is only modified for jets inside that bin, while jets
 * outside keep the nominal scale factors.
 *
 * Definition of a single JER uncertainty bin as provided in the JSON config.
 * The bin is identified by a label and optionally restricted to a region in
 * \f$\eta\f$ and \f$p_T\f$.
 */
struct JerBin {
    std::string label;   // e.g. "CMS_res_j_2017_absEta0to1p93_pT0toInf" or "CMS_res_j_2017"
    double etaMin{}, etaMax{}, ptMin{}, ptMax{};
};

template<typename Specs>
void applyJERNominalOrShift(NanoTree& nanoT,
                  const std::string& year,
                  CorrectionRefs& refs,
                  const std::vector<UInt_t>& idxs,
                  const std::string& var,                          // "nom" / "up" / "down"
                  const std::optional<JerBin>& region = std::nullopt,
                  bool print=false)
{
    if (idxs.empty()) return;
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

        const double reso = refs.corrRefJerReso->evaluate({ etaJet, ptJet, nanoT.Rho });
        double sf = 1.0;
        if(year=="2022Pre" || year=="2022Post" || year=="2023Pre" || year=="2023Post" || year=="2024"){
            sf   = refs.corrRefJerSf->evaluate({ etaJet, ptJet, useVar });
        }else{
            sf   = refs.corrRefJerSf->evaluate({ etaJet, useVar });
        }

        refs.randomGen.SetSeed(nanoT.event + nanoT.run + nanoT.luminosityBlock);

        const int genIdx = Specs::getGenIdx(nanoT, idx);
        bool isMatch = false;
        if (Specs::isValidGenIdx(nanoT, genIdx)) {
            const double dR = deltaR(etaJet, phiJet, Specs::getGenEta(nanoT, genIdx),
                         Specs::getGenPhi(nanoT, genIdx));
            if (dR < Specs::MINDR &&
                std::abs(ptJet - Specs::getGenPt(nanoT, genIdx)) < 3.0 * reso * ptJet) {
                isMatch = true;
            }
        }

        double corr = 1.0;
        if (isMatch) {
            corr = std::max(0.0, 1.0 + (sf - 1.0) * (ptJet - Specs::getGenPt(nanoT, genIdx)) / ptJet);
            printDebug(print, INDENT_JET, "Matched: JER corr =", corr); 
        } else {
            corr = std::max(0.0, 1.0 + refs.randomGen.Gaus(0.0, reso) *
                                     std::sqrt(std::max(sf*sf - 1.0, 0.0)));
            printDebug(print, INDENT_JET, "UnMatched: JER corr =", corr); 
        }
        printDebug(print, INDENT_JET, "[Jet] index=", idx, 
                                          ", eta= ", Specs::getEta(nanoT,idx),
                                          ", isGenMatched = ", isMatch, 
                                          ", smearing factor = ", corr 
                                          );
        printDebug(print, INDENT_DETAIL, "JES corrected  Pt=", Specs::getPt(nanoT,idx));
        Specs::applyCorrection(nanoT, idx, corr);

        std::string extra;
        if (region.has_value()) {
            extra = "   [inRegion=" + std::string(inRegion(*region) ? "yes" : "no") + "]";
        }
        printDebug(print, INDENT_DETAIL, "JER(", useVar, ") smeared Pt=", Specs::getPt(nanoT, idx), extra);

    }
}


// Map<setName, vector<JerBin>>, e.g. "ForUncertaintyJERFull" -> [bins], "ForUncertaintyJERTotal" -> [1 bin]
using JerSetMap = std::map<std::string, std::vector<JerBin>>;

/**
 * Parse the JER uncertainty definitions from the JSON configuration for a
 * given year.
 */
static JerSetMap getJerUncertaintySets(const nlohmann::json& baseJson, const std::string& year) {
    JerSetMap out;

    auto itYear = baseJson.find(year);
    if (itYear == baseJson.end()) return out;
    const auto& y = *itYear;

    auto itUncertainty = y.find("ForUncertaintyJER");
    if (itUncertainty == y.end() || !itUncertainty->is_object()) return out;
    const auto& j = *itUncertainty; // { "ForUncertaintyJERFull": { label: [..], ... }, ... }

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string setName = it.key();  // e.g. "ForUncertaintyJERFull"
        const auto& obj = it.value();
        if (!obj.is_object()) continue;

        std::vector<JerBin> bins;
        bins.reserve(obj.size());
        for (auto it2 = obj.begin(); it2 != obj.end(); ++it2) {
            const std::string label = it2.key();   // e.g. "CMS_res_j_2017_absEta0to1p93_pT0toInf"
            const auto& arr = it2.value();         // expected: [etaMin, etaMax, ptMin, ptMax]

            if (!arr.is_array() || arr.size() != 4) {
                throw std::runtime_error(
                    "ForUncertaintyJER bin \"" + label + "\" must be an array [etaMin, etaMax, ptMin, ptMax]."
                );
            }

            JerBin b;
            b.label = label;
            b.etaMin = arr.at(0).get<double>();
            b.etaMax = arr.at(1).get<double>();
            b.ptMin  = arr.at(2).get<double>();
            b.ptMax  = arr.at(3).get<double>();

            bins.push_back(std::move(b));
        }

        out.emplace(setName, std::move(bins));
    }

    return out;
}

//--------------------------------------------------
// Uncertainty Up/Down JES Correction 
//--------------------------------------------------

// Each entry: { fullTag, base/custom name }
using SystPairJES   = std::pair<std::string, std::string>;
using SystSetMapJES = std::map<std::string, std::vector<SystPairJES>>;

/**
 * Extract the JES systematic tag names for a given year from the JSON
 * configuration.
 */
SystSetMapJES getSystTagNames(const nlohmann::json& baseJson, const std::string& year) {
    SystSetMapJES out;
    if (!baseJson.contains(year)) return out;

    const auto& yearObj = baseJson.at(year);
    if (!yearObj.contains("ForUncertaintyJES")) return out;

    const auto& systs = yearObj.at("ForUncertaintyJES");
    for (auto it = systs.begin(); it != systs.end(); ++it) {
        const std::string setName = it.key();
        const auto& arr = it.value();
        if (!arr.is_array()) continue;

        std::vector<SystPairJES> pairs;
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
 * Description of a single systematic variation to be applied.  
 * For JES systematics it stores the tags for AK4 and AK8 corrections, 
 * while for JER systematics it records the affected `JerBin` region.
 */
struct SystTagDetail {
    std::string setName;   // e.g. "ForUncertaintyJESFull"
    std::string var;       // "Up" / "Down" (empty for nominal)
    SystKind kind{SystKind::Nominal};

    bool isNominal() const { return kind == SystKind::Nominal; }
    std::string systSetName() const { return isNominal() ? "Nominal" : setName; }
    virtual std::string systName() const { return isNominal() ? "Nominal" : setName + "_" + var; }
    virtual ~SystTagDetail() = default;
};

struct SystTagDetailJES : public SystTagDetail {
    std::string tagAK4;    // JES only
    std::string tagAK8;    // JES only
    std::string baseTag;   // CustomName

    std::string systName() const override { return isNominal() ? "Nominal" : baseTag + "_" + var; }
};

struct SystTagDetailJER : public SystTagDetail {
    std::string baseTag;   // label (e.g. "CMS_res_j_2017_absEta…")
    JerBin jerRegion;      // region definition from JSON

    std::string systName() const override { return isNominal() ? "Nominal" : baseTag + "_" + var; }
};

/**
 * Expand the JER uncertainty definition into explicit up/down variations for
 * each bin defined in the JSON configuration.
 */
static std::vector<SystTagDetailJER> buildJerTagDetails(const JerSetMap& jerSets) {
    std::vector<SystTagDetailJER> out;
    for (const auto& [setName, bins] : jerSets) {
        for (const auto& b : bins) {
            for (const char* var : {"Up","Down"}) {
                SystTagDetailJER d;
                d.setName   = setName;        // "ForUncertaintyJERFull" or "ForUncertaintyJERTotal"
                d.var       = var;
                d.kind      = SystKind::JER;
                d.baseTag   = b.label;        // the label/key from JSON
                d.jerRegion = b;
                out.push_back(d);
            }
        }
    }
    return out;
}

/**
 * Combine the JES systematic tag information for AK4 and AK8 jets.  
 * Only systematics defined for both algorithms are returned.
 */
static std::vector<SystTagDetailJES> buildSystTagDetailJES(const SystSetMapJES& sAK4,
                                                      const SystSetMapJES& sAK8)
{
    std::vector<SystTagDetailJES> systTagDetails;

    for (const auto& [set, pairsAK4] : sAK4) {
        auto itAK8 = sAK8.find(set);
        if (itAK8 == sAK8.end()) continue;  // set must exist for both algos
        const auto& pairsAK8 = itAK8->second;

        // Map base/custom name -> fullTag for each algo
        std::unordered_map<std::string, std::string> mapAK4, mapAK8;
        mapAK4.reserve(pairsAK4.size());
        mapAK8.reserve(pairsAK8.size());

        for (const auto& p : pairsAK4) mapAK4.emplace(p.second, p.first); // key = base/custom
        for (const auto& p : pairsAK8) mapAK8.emplace(p.second, p.first);

        // Intersect on custom name (2nd element)
        for (const auto& [base, fullAK4] : mapAK4) {
            auto itFullAK8 = mapAK8.find(base);
            if (itFullAK8 == mapAK8.end()) continue;

            for (const char* var : {"Up","Down"}) {
                SystTagDetailJES d;
                d.setName = set;
                d.var = var;
                d.kind = SystKind::JES;
                d.tagAK4 = fullAK4;
                d.tagAK8 = itFullAK8->second;
                d.baseTag = base;
                systTagDetails.push_back(d);
            }
        }
    }
    return systTagDetails;
}

/**
 * Apply a JES systematic variation.  The specific correction is selected by
 * \p systName and scaled up or down according to \p var.
 */
template<typename Specs>
void applySystShiftJES(NanoTree& nanoT,
                          CorrectionRefs& refs,
                          const std::string& systName,
                          const std::string& var,
                          const std::vector<UInt_t>& idxs,
                          bool print=false
                          )
{
    auto systCorr = safeGet(refs.cs, systName);
    for(auto idx: idxs){
        printDebug(print, INDENT_JET, "[Jet] index=", idx);
        printDebug(print, INDENT_DETAIL, "Nominal corrected    Pt=", Specs::getPt(nanoT,idx));
        double scale = systCorr->evaluate({ Specs::getEta(nanoT,idx), Specs::getPt(nanoT,idx) });
        double shift = (var=="Up" ? 1+scale : 1-scale);
        Specs::applyCorrection(nanoT, idx, shift);
        printDebug(print, INDENT_DETAIL, "Syst corrected    Pt=", Specs::getPt(nanoT,idx));

    }
}

// Propagate the effect of jet corrections to MET using the stored raw jet pT.
// The JES and JER are applied starting from the muon-subtracted raw jet pT and
// the difference with respect to the L1 corrected value is propagated to MET.
// The index list must correspond to the raw pT bookkeeping (i.e. use the
// original NanoAOD ordering).
TLorentzVector getCorrectedMet(NanoTree& nanoT,
                       const std::string& year,
                       CorrectionRefs& refs,
                       bool isData,
                       const std::vector<UInt_t>& idxs,
                       const std::vector<double>& rawPts,
                       const std::string& jerVar = "nom",
                       const std::optional<JerBin>& jerRegion = std::nullopt,
                       const std::string& jesSystName = "",
                       const std::string& jesSystVar = "", bool print=true) {

    double met_raw_pt, met_raw_phi;
    if(usesPuppiMet(year)){
        met_raw_pt  = nanoT.RawPuppiMET_pt;
        met_raw_phi = nanoT.RawPuppiMET_phi;
    }else{
        met_raw_pt  = nanoT.RawMET_pt;
        met_raw_phi = nanoT.RawMET_phi;
    }
    double met_px = met_raw_pt * std::cos(met_raw_phi);
    double met_py = met_raw_pt * std::sin(met_raw_phi);
    printDebug(print, INDENT_BLOCK, "[Met] Raw Pt =", met_raw_pt);
    for (const auto idx : idxs) {
        if (idx >= rawPts.size()) continue; // guard against mismatched bookkeeping
        const double phi  = nanoT.Jet_phi[idx];
        const double eta  = nanoT.Jet_eta[idx];
        const double area = nanoT.Jet_area[idx];

        // Recompute corrections on muon-subtracted raw jet pT.
        const double pt_raw_minusMuon = rawPts[idx] * (1 - nanoT.Jet_muonSubtrFactor[idx]);
        double pt_corr   = pt_raw_minusMuon;
        // L1FastJet (pileup correction).
        double c1 = refs.corrRefJesL1FastJet->evaluate({area, eta, pt_corr, nanoT.Rho});
        pt_corr   *= c1;
        // Record the pT immediately after the L1 step.
        double pt_corr_l1rc = pt_corr;

        // L2Relative (MC truth correction).
        double c2 = 1.0;
        if(hasPhiDependentL2(year)){
            c2 = refs.corrRefJesL2Relative->evaluate({ eta, phi, pt_corr });
        }else{
            c2 = refs.corrRefJesL2Relative->evaluate({ eta, pt_corr });
        }
        pt_corr   *= c2;

        // L2L3Residual (L2Residual + L3Residual together) applied only to data.
        // (Covers residual differences between data and simulation.)
        if(isData){
          double cR = 1.0;
          if(requiresRunBasedResidual(year)){
            const double runNumber = representativeRunNumber(nanoT, year);
            cR = refs.corrRefJesL2ResL3Res->evaluate({runNumber, eta, pt_corr });
          }else{
            cR = refs.corrRefJesL2ResL3Res->evaluate({ eta, pt_corr });
          }
          pt_corr   *= cR;
        }

        // JES systematic shift if requested (MC only).
        if(!isData && !jesSystName.empty()){
            auto systCorr = safeGet(refs.cs, jesSystName);
            double scale = systCorr->evaluate({ eta, pt_corr });
            double shift = (jesSystVar == "Up" ? 1 + scale : 1 - scale);
            pt_corr *= shift;
        }

        // JER smearing (MC only).
        if(!isData){
            std::string useVar = jerVar;
            if(jerRegion.has_value()){
                const double aeta = std::fabs(eta);
                if(!(aeta >= jerRegion->etaMin && aeta < jerRegion->etaMax &&
                     pt_corr >= jerRegion->ptMin && pt_corr < jerRegion->ptMax)){
                    useVar = "nom";
                }
            }

            const double reso = refs.corrRefJerReso->evaluate({ eta, pt_corr, nanoT.Rho });
            double sf = 1.0;
            if(usesPuppiMet(year)){
                sf = refs.corrRefJerSf->evaluate({ eta, pt_corr, useVar });
            }else{
                sf = refs.corrRefJerSf->evaluate({ eta, useVar });
            }

            refs.randomGen.SetSeed(nanoT.event + nanoT.run + nanoT.luminosityBlock);

            const int genIdx = nanoT.Jet_genJetIdx[idx];
            bool isMatch = false;
            if(genIdx > -1 && static_cast<UInt_t>(genIdx) < nanoT.nGenJet){
                const double dR = deltaR(eta, phi, 
                                        nanoT.GenJet_eta[genIdx], nanoT.GenJet_phi[genIdx]);
                if(dR < AK4Specs::MINDR &&
                   std::abs(pt_corr - nanoT.GenJet_pt[genIdx]) < 3.0 * reso * pt_corr){
                    isMatch = true;
                }
            }

            double corr = 1.0;
            if(isMatch){
                corr = std::max(0.0, 1.0 + (sf - 1.0) *
                                         (pt_corr - nanoT.GenJet_pt[genIdx]) / pt_corr);
            } else {
                corr = std::max(0.0, 1.0 + refs.randomGen.Gaus(0.0, reso) *
                                         std::sqrt(std::max(sf*sf - 1.0, 0.0)));
            }
            pt_corr *= corr;
        }

        // Selection used when propagating to MET.
        const bool passSel = (pt_corr > 15.0
                             && std::abs(eta) < 5.2
                             && (nanoT.Jet_chEmEF[idx] + nanoT.Jet_neEmEF[idx]) < 0.9
                             );
        if (!passSel) continue;
        printDebug(print, INDENT_JET, "[Jet] index=", idx);
        printDebug(print, INDENT_STEP, "L1 JEC corrected Pt =", pt_corr_l1rc);
        printDebug(print, INDENT_STEP, "L1L2L3JEC + JER corrected Pt =", pt_corr);
        const double dpt = (pt_corr - pt_corr_l1rc);
        met_px -= dpt * std::cos(phi);
        met_py -= dpt * std::sin(phi);
    }
    // Finalise MET.
    const double met_pt  = std::hypot(met_px, met_py);
    const double met_phi = std::atan2(met_py, met_px);
    printDebug(print, INDENT_BLOCK, "[Met] Type-1 corrected Pt =", met_pt);
    TLorentzVector p4MetCorr;
    p4MetCorr.SetPtEtaPhiM(met_pt, 0, met_phi, 0);
    return p4MetCorr;
}


//--------------------------------------------------
// Jet Veto Map
//--------------------------------------------------
/**
 * Check whether any reconstructed jet falls inside a jet veto region as
 * defined by the provided correction map.
 */
bool checkIfAnyJetInVetoRegion(const correction::Correction::Ref &jvmRef,
                               const std::string& jvmKeyName,
                               const NanoTree& nanoT){
    const double maxEtaInMap = 5.191;
    const double maxPhiInMap = 3.1415926;
    bool vetoEvent = false;
    for (int i = 0; i != nanoT.nJet; ++i) {
        // Apply a minimal jet selection.
        if (std::abs(nanoT.Jet_eta[i]) > maxEtaInMap) continue;
        if (std::abs(nanoT.Jet_phi[i]) > maxPhiInMap) continue;
        if (nanoT.Jet_jetId[i] < 6 ) continue;// TightLepVeto ID.
        if (nanoT.Jet_pt[i] < 15) continue;
        if ((nanoT.Jet_chEmEF[i] + nanoT.Jet_neEmEF[i]) > 0.9) continue;

        // Check whether the jet lies inside the veto region.
        auto jvmNumber = jvmRef->evaluate({jvmKeyName, nanoT.Jet_eta[i], nanoT.Jet_phi[i]});
        // The correction returns 0.0 outside the veto region and 100.0 inside it.
        if (jvmNumber > 0.0) {
            vetoEvent = true;
            break; // No need to loop over the remaining jets.
        }
    }// nJet loop

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
 * Fill a histogram only if the pointer is non-null.  This avoids repeated
 * null checks at the call site and centralises the behaviour in a single
 * function.
 */
inline void fillHist(TH1F* h, double x) {
    if (h) h->Fill(x);
}


static TDirectory* getOrMkdir(TDirectory* parent, const std::string& name) {
    if (!parent) return nullptr;
    if (auto* d = dynamic_cast<TDirectory*>(parent->Get(name.c_str()))) return d;
    return parent->mkdir(name.c_str());
}

/**
 * Create and organise the monitoring histograms in the output file.  The
 * directory structure is `year/type/systematicSet/systematicName`.  For data
 * samples the optional `era` is inserted between `type` and `systematicSet`
 * to avoid histogram name clashes across eras.
 */
static Hists makeHists(TFile& fout,
                       const std::string& year,
                       bool isData,
                       const std::optional<std::string>& era,
                       const std::string& systSetName,
                       const std::string& systName,
                       bool alsoNano = false)
{
    auto* yearDir = dynamic_cast<TDirectory*>(fout.Get(year.c_str()));
    if (!yearDir) yearDir = fout.mkdir(year.c_str());

    TDirectory* typeDir = getOrMkdir(yearDir, isData ? "Data" : "MC");
    if (isData && era) {
        typeDir = getOrMkdir(typeDir, *era);
    }
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
// Jet selection helpers
//--------------------------------------------------
/**
 * Collect indices of AK8 jets passing basic kinematic requirements.  The
 * NanoAOD values are optionally filled into the monitoring histogram.
 */
std::vector<UInt_t> collectAK8Jets(const NanoTree& nanoT, TH1F* hNano) {
    std::vector<UInt_t> idxs;
    idxs.reserve(nanoT.nFatJet);
    for (UInt_t j = 0; j < nanoT.nFatJet; ++j) {
        if (nanoT.FatJet_pt[j] < 100 || std::abs(nanoT.FatJet_eta[j]) > 5.2)
            continue;
        idxs.push_back(j);
        fillHist(hNano, nanoT.FatJet_pt[j]);
    }
    return idxs;
}

/**
 * Collect indices of AK4 jets that do not overlap with any selected AK8 jet.
 * Jets failing the basic kinematic requirements are discarded.  If provided, a
 * histogram of the NanoAOD jet \f$p_T\f$ is filled.
 */
std::vector<UInt_t> collectNonOverlappingAK4Jets(const NanoTree& nanoT,
                                   const std::vector<UInt_t>& ak8Idxs,
                                   TH1F* hNano) {
    std::vector<UInt_t> idxs;
    idxs.reserve(nanoT.nJet);
    for (UInt_t j = 0; j < nanoT.nJet; ++j) {
        if (nanoT.Jet_pt[j] < 15 || std::abs(nanoT.Jet_eta[j]) > 5.2)
            continue;
        bool overlaps = false;
        for (auto k : ak8Idxs) {
            if (deltaR(nanoT.Jet_eta[j], nanoT.Jet_phi[j],
                       nanoT.FatJet_eta[k], nanoT.FatJet_phi[k]) < 0.6) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) continue;
        idxs.push_back(j);
        fillHist(hNano, nanoT.Jet_pt[j]);
    }
    return idxs;
}

//--------------------------------------------------
// Events are looped in this function
//--------------------------------------------------
/**
 * Core event loop applying the nominal corrections and, depending on
 * \c systTagDetail, an additional JES or JER systematic variation.  Histograms
 * are filled and written to the provided output file.
 */
static void processEvents(TChain& chain,
                              NanoTree& nanoT,
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

    // --- AK8 refs: reuse AK4 tags if AK8 lacks this year (Run-3)
    auto cfgAK8  = loadJsonConfig("JercFileAndTagNamesAK8.json");
    Tags tagsAK8;
    if (cfgAK8.contains(year)) {
        tagsAK8 = getTagNames(cfgAK8, year, isData, era);
    } else {// Use the AK4 JERC for AK8
        tagsAK8 = tagsAK4;
    }
    CorrectionRefs refsAK8(tagsAK8);

    // --- JVM 
    auto cfgJvm   = loadJsonConfig("JvmFileAndTagNames.json");
    correction::Correction::Ref jvmRef;
    std::string jvmKey;
    bool useJvm = false;
    if (cfgJvm.contains(year)) {
        const auto& y = cfgJvm.at(year);
        const auto& jvmFile = getTagName(y, "jvmFilePath");
        const auto& jvmTag  = getTagName(y, "jvmTagName");
        jvmRef  = correction::CorrectionSet::from_file(jvmFile)->at(jvmTag);
        jvmKey  = getTagName(y, "jvmKeyName");
        useJvm  = true;
    }

    // Determine once which jet collections need to be corrected in this pass.
    const bool runAK4 = applyOnlyOnAK4 || applyOnAK4AndAK8;
    const bool runAK8 = applyOnlyOnAK8 || applyOnAK4AndAK8;

    const bool writeNano = systTagDetail.isNominal(); // write once in Nominal pass
    Hists H = makeHists(fout, year, isData, era, systTagDetail.systSetName(),
                        systTagDetail.systName(), writeNano);

    const Long64_t nEntries = chain.GetEntries();
    int countVeto = 0;
    int printCount = 0;
    bool print = false;
    for (Long64_t i = 0; i < nEntries; ++i) {
        if (chain.LoadTree(i) < 0) break;
        chain.GetTree()->GetEntry(i);

        fillHist(H.hMET_Nano, nanoT.MET_pt);

        // --- Select jet indices using helper functions.
        std::vector<UInt_t> indicesAK8;
        if (runAK8) {
            indicesAK8 = collectAK8Jets(nanoT, H.hJetPt_AK8_Nano);
        }
        std::vector<UInt_t> indicesAK4;
        if (runAK4) {
            indicesAK4 = collectNonOverlappingAK4Jets(nanoT, indicesAK8, H.hJetPt_AK4_Nano);
        }

        // Store raw AK4 jet pT values for MET propagation (only AK4 contributes to MET).
        const int nAk4Jets = std::max(nanoT.nJet, 0);
        std::vector<double> rawPtsAK4ForMet(static_cast<size_t>(nAk4Jets));
        std::vector<UInt_t> jetIndicesForMet;
        jetIndicesForMet.reserve(static_cast<size_t>(nAk4Jets));
        // Keep the raw jet ordering when storing indices for MET propagation.
        for (int i = 0; i < nAk4Jets; ++i){
            rawPtsAK4ForMet[i] = (1 - nanoT.Jet_rawFactor[i]) * nanoT.Jet_pt[i];
            jetIndicesForMet.push_back(static_cast<UInt_t>(i));
        }

        bool whenToPrint = false;
        if(runAK4 && runAK8){
            whenToPrint = !indicesAK4.empty() && !indicesAK8.empty();
        }else if(runAK4){
            whenToPrint = !indicesAK4.empty();
        }else if(runAK8){
            whenToPrint = !indicesAK8.empty();
        }
        if(whenToPrint && printCount==0){
            printCount++; print = true;
        }

        // =========================
        // 1) JES (nominal) — apply only the nominal JES.
        // =========================
        if (systTagDetail.isNominal()) {
            // In the nominal pass print the full per-jet breakdown.
            if (runAK4) {
                printDebug(print, INDENT_BLOCK, "AK4 (JES nominal) on "+std::to_string(indicesAK4.size())+" jets");
                applyJESNominal<AK4Specs>(nanoT, year, refsAK4, isData, indicesAK4, print);
            }

            if (runAK8) {
                printDebug(print, INDENT_BLOCK, "AK8 (JES nominal) on "+std::to_string(indicesAK8.size())+" jets");
                applyJESNominal<AK8Specs>(nanoT, year, refsAK8, isData, indicesAK8, print);
            }
        } else {
            // For systematic shifts we still need to apply the nominal
            // corrections but skip the verbose printing to avoid repeating
            // the same information for each shift.
            if (runAK4) {
                applyJESNominal<AK4Specs>(nanoT, year, refsAK4, isData, indicesAK4, false);
            }
            if (runAK8) {
                applyJESNominal<AK8Specs>(nanoT, year, refsAK8, isData, indicesAK8, false);
            }
            printDebug(print, INDENT_BLOCK, "Nominal JES applied");
        }

        // =========================
        // 2) JES uncertainty (MC only) when running a JES systematic pass.
        // =========================
        if (!isData && systTagDetail.kind == SystKind::JES) {
            const auto& jesDetail = static_cast<const SystTagDetailJES&>(systTagDetail);
            if (runAK4) {
                printDebug(print, INDENT_BLOCK, "AK4 (JES ", jesDetail.var, ")");
                applySystShiftJES<AK4Specs>(nanoT, refsAK4, jesDetail.tagAK4, jesDetail.var, indicesAK4, print);
            }

            if (runAK8) {
                printDebug(print, INDENT_BLOCK, "AK8 (JES ", jesDetail.var, ")");
                applySystShiftJES<AK8Specs>(nanoT, refsAK8, jesDetail.tagAK8, jesDetail.var, indicesAK8, print);
            }
        }

        // Determine which JES/JER variations need to be propagated to MET.
        std::string jerVar = "nom";
        std::optional<JerBin> jerRegion;
        std::string jesSystName;
        std::string jesVar;
        if (!isData) {
            if (systTagDetail.kind == SystKind::JES) {
                const auto& jesDetail = static_cast<const SystTagDetailJES&>(systTagDetail);
                jesSystName = jesDetail.tagAK4;
                jesVar      = jesDetail.var;
            } else if (systTagDetail.kind == SystKind::JER) {
                const auto& jerDetail = static_cast<const SystTagDetailJER&>(systTagDetail);
                jerVar    = (jerDetail.var == "Up" ? "up" : "down");
                jerRegion = jerDetail.jerRegion;
            }
        }

        // =========================
        // 3) JER (after all JES): apply nominal or shifted smearing for MC.
        // =========================
        if (!isData) {
            if (runAK4) {
                printDebug(print, INDENT_BLOCK, "AK4 (JER ", jerVar == "nom" ? "nom" : (jerVar == "up" ? "Up" : "Down"), ")");
                applyJERNominalOrShift<AK4Specs>(nanoT, year, refsAK4, indicesAK4, jerVar, jerRegion, print);
            }
            if (runAK8) {
                printDebug(print, INDENT_BLOCK, "AK8 (JER ", jerVar == "nom" ? "nom" : (jerVar == "up" ? "Up" : "Down"), ")");
                applyJERNominalOrShift<AK8Specs>(nanoT, year, refsAK8, indicesAK8, jerVar, jerRegion, print);
            }
        }

        if (applyOnMET) {
            TLorentzVector p4CorrectedMET = getCorrectedMet(nanoT, year, refsAK4, isData,
                                                            jetIndicesForMet, rawPtsAK4ForMet,
                                                            jerVar, jerRegion, jesSystName, jesVar, print);
            H.hMET->Fill(p4CorrectedMET.Pt());
        }

        // Fill histograms with corrected jet pT.
        for (auto idx : indicesAK4) H.hJetPt_AK4->Fill(nanoT.Jet_pt[idx]);
        for (auto idx : indicesAK8) H.hJetPt_AK8->Fill(nanoT.FatJet_pt[idx]);

        print = false;


        // =========================
        // 4) Jet veto map check.
        // =========================
        if (useJvm && checkIfAnyJetInVetoRegion(jvmRef, jvmKey, nanoT)) {
            countVeto++;
            continue;
        }

        // Additional analysis steps can be inserted here.
    }//event loop

    std::cout<<"   \nNumber of events vetoed due to JetVetoMap: "<<countVeto<<'\n';
}


/**
 * Helper that runs the core event loop for the nominal pass and for all
 * requested JES and JER systematic variations for a given year.  Systematic
 * definitions are retrieved from the JSON configuration files.
 *
 * NOTE: This function now creates a *fresh* NanoTree and rebinds branches
 * for every pass (Nominal / each JES / each JER), so corrections never pile up.
 */
void processEventsWithNominalOrSyst(TChain& chain,
                                    TFile& fout,
                                    const std::string& year,
                                    bool isData,
                                    const std::optional<std::string>& era = std::nullopt)
{
    auto cfgAK4 = loadJsonConfig("JercFileAndTagNamesAK4.json");
    auto cfgAK8 = loadJsonConfig("JercFileAndTagNamesAK8.json");

    // JES sets: if AK8 tags are missing for Run-3, fall back to AK4 tags
    SystSetMapJES sAK4 = getSystTagNames(cfgAK4, year);
    SystSetMapJES sAK8;
    if (cfgAK8.contains(year)) {
        sAK8 = getSystTagNames(cfgAK8, year);
    } else {
        sAK8 = sAK4; // use AK4 tags for AK8 when not available
    }

    // JER sets (read once; we can use AK4 file as the source of binning)
    JerSetMap jerSets = getJerUncertaintySets(cfgAK4, year);

    std::vector<SystTagDetailJES> jesDetails;
    std::vector<SystTagDetailJER> jerDetails;
    size_t totalPasses = 1; // Nominal pass is always run

    if (!isData) {
        if (applyOnlyOnAK4) {
            for (const auto& [set, pairs] : sAK4) {
                for (const auto& p : pairs) {
                    for (const char* var : {"Up", "Down"}) {
                        SystTagDetailJES d;
                        d.setName = set;
                        d.var = var;
                        d.kind = SystKind::JES;
                        d.tagAK4 = p.first;
                        d.tagAK8.clear();
                        d.baseTag = p.second;
                        jesDetails.push_back(d);
                    }
                }
            }
        } else if (applyOnlyOnAK8) {
            for (const auto& [set, pairs] : sAK8) {
                for (const auto& p : pairs) {
                    for (const char* var : {"Up", "Down"}) {
                        SystTagDetailJES d;
                        d.setName = set;
                        d.var = var;
                        d.kind = SystKind::JES;
                        d.tagAK4.clear();
                        d.tagAK8 = p.first;
                        d.baseTag = p.second;
                        jesDetails.push_back(d);
                    }
                }
            }
        } else {
            jesDetails = buildSystTagDetailJES(sAK4, sAK8);
        }
        totalPasses += jesDetails.size();

        jerDetails = buildJerTagDetails(jerSets);
        totalPasses += jerDetails.size();
    }

    std::string progressPrefix = isData ? "Data " : "MC ";
    progressPrefix += year;
    if (isData && era) {
        progressPrefix += " (" + *era + ")";
    }

    // Use stderr to keep logs and progress separated
    boost::timer::progress_display progress(totalPasses, std::cerr);
    std::cerr << "\n" << progressPrefix << " : running " << totalPasses << " passes...\n";
    auto run_pass = [&](const SystTagDetail& detail, const char* banner = nullptr) {
        if (banner) std::cout << banner << '\n';  // same banner printing as before
        NanoTree nanoT;
        setupNanoBranches(&chain, nanoT, isData);
        processEvents(chain, nanoT, fout, year, isData, era, detail);
        ++progress;  // one tick per finished pass
    };


    // 0) Nominal pass.
    run_pass(SystTagDetail{}, " [Nominal]");

    if (!isData) {
        // 1) Correlated JES systematics.
        for (const auto& d : jesDetails) {
            std::cout << "\n [JES Syst]: " << d.systName() << '\n';
            run_pass(d);
        }

        // 2) JER systematics (Up/Down) with region gating from JSON.
        for (const auto& d : jerDetails) {
            std::cout << "\n [JER Syst]: " << d.systSetName() << "/" << d.systName() << '\n';
            run_pass(d);
        }
    }
}



/**
 * Macro entry point.  Configures the input files and years to process and
 * orchestrates running over MC and data samples.
 */
void applyJercAndJvm() {
    const std::string fInputData = "NanoAOD_Data.root";
    const std::string fInputMc   = "NanoAOD_MC.root";

    // Prepare the output file.
    std::string outName = "output_C.root";
    TFile fout(outName.c_str(), "RECREATE");

    std::vector<std::string> mcYears = {
        "2016Pre", 
        //"2016Post",
        //"2017",
        //"2018",
        //"2022Pre",
        //"2022Post",
        //"2023Pre",
        //"2023Post",
        //"2024"
    };
    std::vector<std::pair<std::string, std::string>> dataConfigs = {
       {"2016Pre", "Era2016PreBCD"},
      //{"2016Pre", "Era2016PreEF"},
      //{"2016Post","Era2016PostFGH"},
      //{"2017",    "Era2017B"},
      //{"2017",    "Era2017C"},
      //{"2017",    "Era2017D"},
      //{"2017",    "Era2017E"},
      //{"2017",    "Era2017F"},
      //{"2018",    "Era2018A"},
      //{"2018",    "Era2018B"},
      //{"2018",    "Era2018C"},
      //{"2018",    "Era2018D"},
      //{"2022Pre", "Era2022C"},
      //{"2022Pre", "Era2022D"},
      //{"2022Post","Era2022E"},
      //{"2022Post","Era2022F"},
      //{"2022Post","Era2022G"},
      //{"2023Pre", "Era2023PreAll"}, // Run based
      //{"2023Post","Era2023PostAll"},// Run based
      //{"2024",    "Era2024All"},    // Run based
    };

    for (const auto& year : mcYears) {
        std::cout<<"-----------------\n";
        std::cout<<"[MC] : "<<year <<'\n';
        std::cout<<"-----------------\n";
        TChain chain("Events");
        chain.Add(fInputMc.c_str());
        processEventsWithNominalOrSyst(chain, fout, year, /*isData=*/false);
    }

    for (const auto& [year, era] : dataConfigs) {
        std::cout<<"-----------------\n";
        std::cout<<"\n[Data] : "<<year <<" : "<<era <<'\n';
        std::cout<<"-----------------\n";
        TChain chain("Events");
        chain.Add(fInputData.c_str());
        processEventsWithNominalOrSyst(chain, fout, year, /*isData=*/true, era);
    }

    fout.Write();
    fout.Close();
    std::cout << "Wrote output to " << outName << "\n";
}

