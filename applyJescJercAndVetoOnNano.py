#!/usr/bin/env python3
import sys
import json
import math
import random
from enum import IntEnum

import ROOT
import correctionlib._core as correction

#------------------------------------------------------------------------------
# ReadConfig: load JSON config and retrieve values
#------------------------------------------------------------------------------
class ReadConfig:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            self.config = json.load(f)

    def get_value(self, keys):
        current = self.config
        for key in keys:
            if key not in current:
                raise RuntimeError(f"Missing JSON element: {self.filename} -> {key}")
            current = current[key]
        return current

#------------------------------------------------------------------------------
# NanoTree: wrapper for a TChain to read NanoAOD
#------------------------------------------------------------------------------
class NanoTree:
    def __init__(self):
        self.chain = ROOT.TChain("Events")

    def load_tree(self, path):
        print(f"==> loadTree: {path}")
        self.chain.SetCacheSize(100 * 1024 * 1024)
        self.chain.Add(path)
        print(f"Entries: {self.chain.GetEntries()}")

    def get_entries(self):
        return self.chain.GetEntries()

    def get_entry(self, i):
        return self.chain.GetEntry(i)

#------------------------------------------------------------------------------
# ScaleJetMet: apply JECs/JER and print values at each step
#------------------------------------------------------------------------------
class CorrectionLevel(IntEnum):
    NONE        = 0
    L1RC        = 1
    L2REL       = 2
    L2RES_L3RES = 3

class ScaleJetMet:
    def __init__(self):
        self.isMC   = True
        self.isData = False
        self.randomGen = random.Random()
        self._load_config()
        self._load_refs()

    def _load_config(self):
        cfg = ReadConfig("JescJercAndVetoTagName.json")
        year = "2018"
        self.jerc_path       = cfg.get_value([year, "jercJsonPath"])
        self.name_l1         = cfg.get_value([year, "jetL1FastJetName"])
        self.name_l2         = cfg.get_value([year, "jetL2RelativeName"])
        self.name_l2resl3res = cfg.get_value([year, "jetL2L3ResidualName"])
        self.name_jr         = cfg.get_value([year, "JerResoName"])
        self.name_js         = cfg.get_value([year, "JerSfName"])

    def _load_refs(self):
        cs = correction.CorrectionSet.from_file(self.jerc_path)
        self.corrL1      = cs[self.name_l1]
        self.corrL2      = cs[self.name_l2]
        self.corrL2ResL3 = cs[self.name_l2resl3res]
        self.corrJerReso = cs[self.name_jr]
        self.corrJerSf   = cs[self.name_js]

    def apply_corrections(self, nano: NanoTree, level: CorrectionLevel):
        tree = nano.chain

        # MET from NanoAOD
        met_nano = ROOT.TLorentzVector()
        met_nano.SetPtEtaPhiM(tree.ChsMET_pt, 0.0, tree.ChsMET_phi, 0.0)
        print(f"[MET Nano]    Pt = {met_nano.Pt():.3f}  Phi = {met_nano.Phi():.3f}")

        met = ROOT.TLorentzVector(met_nano)

        for i in range(int(tree.nJet)):
            pt0   = tree.Jet_pt[i]
            eta   = tree.Jet_eta[i]
            phi   = tree.Jet_phi[i]
            mass0 = tree.Jet_mass[i]

            if pt0 < 15 or abs(eta) > 5.2:
                continue

            # — Nano
            p4_pt   = pt0
            p4_mass = mass0
            p4 = ROOT.TLorentzVector()
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            met += p4
            print(f"\nJet[{i}] Nano    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — Raw
            raw_scale = 1.0 - tree.Jet_rawFactor[i]
            p4_pt   *= raw_scale
            p4_mass *= raw_scale
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            print(f"Jet[{i}] Raw     Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L1RC
            if level >= CorrectionLevel.L1RC:
                c1 = self.corrL1.evaluate(
                    p4_pt,
                    eta,
                    tree.Jet_area[i],
                    tree.fixedGridRhoFastjetAll
                )
                p4_pt   *= c1
                p4_mass *= c1
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L1Rc    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L2Rel
            if level >= CorrectionLevel.L2REL:
                c2 = self.corrL2.evaluate(
                    p4_pt,
                    eta
                )
                p4_pt   *= c2
                p4_mass *= c2
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L2Rel   Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L2ResL3Res (data only)
            if self.isData and level >= CorrectionLevel.L2RES_L3RES:
                cr = self.corrL2ResL3.evaluate(
                    p4_pt,
                    eta
                )
                p4_pt   *= cr
                p4_mass *= cr
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L2ResL3 Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — JER smearing (MC only)
            if self.isMC:
                reso = self.corrJerReso.evaluate(
                    p4_pt,
                    eta,
                    tree.fixedGridRhoFastjetAll
                )
                sf = self.corrJerSf.evaluate(
                    eta,
                    "nom"
                )
                self.randomGen.seed(tree.event + tree.run + tree.luminosityBlock)
                smear = max(
                    0.0,
                    1.0 + self.randomGen.gauss(0.0, reso)
                            * math.sqrt(max(sf*sf - 1.0, 0.0))
                )
                p4_pt   *= smear
                p4_mass *= smear
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] Jer     Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — Final Corr
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            print(f"Jet[{i}] Corr    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            met -= p4

        print(f"\n[MET Corr]    Pt = {met.Pt():.3f}  Phi = {met.Phi():.3f}\n")
        tree.ChsMET_pt  = met.Pt()
        tree.ChsMET_phi = met.Phi()


def main(input_file: str):
    nano = NanoTree()
    nano.load_tree(input_file)

    scale = ScaleJetMet()
    n_events = nano.get_entries()
    print(f"Events: {n_events}\n")

    for i in range(min(n_events, 10)):
        nano.get_entry(i)
        scale.apply_corrections(nano, CorrectionLevel.L2RES_L3RES)
        print("="*20 + f" End of Event {i} " + "="*20 + "\n")

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "NanoAod.root"
    main(infile)
#!/usr/bin/env python3
import sys
import json
import math
import random
from enum import IntEnum

import ROOT
import correctionlib._core as correction

#------------------------------------------------------------------------------
# ReadConfig: load JSON config and retrieve values
#------------------------------------------------------------------------------
class ReadConfig:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            self.config = json.load(f)

    def get_value(self, keys):
        current = self.config
        for key in keys:
            if key not in current:
                raise RuntimeError(f"Missing JSON element: {self.filename} -> {key}")
            current = current[key]
        return current

#------------------------------------------------------------------------------
# NanoTree: wrapper for a TChain to read NanoAOD
#------------------------------------------------------------------------------
class NanoTree:
    def __init__(self):
        self.chain = ROOT.TChain("Events")

    def load_tree(self, path):
        print(f"==> loadTree: {path}")
        self.chain.SetCacheSize(100 * 1024 * 1024)
        self.chain.Add(path)
        print(f"Entries: {self.chain.GetEntries()}")

    def get_entries(self):
        return self.chain.GetEntries()

    def get_entry(self, i):
        return self.chain.GetEntry(i)

#------------------------------------------------------------------------------
# ScaleJetMet: apply JECs/JER and print values at each step
#------------------------------------------------------------------------------
class CorrectionLevel(IntEnum):
    NONE        = 0
    L1RC        = 1
    L2REL       = 2
    L2RES_L3RES = 3

class ScaleJetMet:
    def __init__(self):
        self.isMC   = True
        self.isData = False
        self.randomGen = random.Random()
        self._load_config()
        self._load_refs()

    def _load_config(self):
        cfg = ReadConfig("JescJercAndVetoTagName.json")
        year = "2018"
        self.jerc_path       = cfg.get_value([year, "jercJsonPath"])
        self.name_l1         = cfg.get_value([year, "jetL1FastJetName"])
        self.name_l2         = cfg.get_value([year, "jetL2RelativeName"])
        self.name_l2resl3res = cfg.get_value([year, "jetL2L3ResidualName"])
        self.name_jr         = cfg.get_value([year, "JerResoName"])
        self.name_js         = cfg.get_value([year, "JerSfName"])

    def _load_refs(self):
        cs = correction.CorrectionSet.from_file(self.jerc_path)
        self.corrL1      = cs[self.name_l1]
        self.corrL2      = cs[self.name_l2]
        self.corrL2ResL3 = cs[self.name_l2resl3res]
        self.corrJerReso = cs[self.name_jr]
        self.corrJerSf   = cs[self.name_js]

    def apply_corrections(self, nano: NanoTree, level: CorrectionLevel):
        tree = nano.chain

        # MET from NanoAOD
        met_nano = ROOT.TLorentzVector()
        met_nano.SetPtEtaPhiM(tree.ChsMET_pt, 0.0, tree.ChsMET_phi, 0.0)
        print(f"[MET Nano]    Pt = {met_nano.Pt():.3f}  Phi = {met_nano.Phi():.3f}")

        met = ROOT.TLorentzVector(met_nano)

        for i in range(int(tree.nJet)):
            pt0   = tree.Jet_pt[i]
            eta   = tree.Jet_eta[i]
            phi   = tree.Jet_phi[i]
            mass0 = tree.Jet_mass[i]

            if pt0 < 15 or abs(eta) > 5.2:
                continue

            # — Nano
            p4_pt   = pt0
            p4_mass = mass0
            p4 = ROOT.TLorentzVector()
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            met += p4
            print(f"\nJet[{i}] Nano    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — Raw
            raw_scale = 1.0 - tree.Jet_rawFactor[i]
            p4_pt   *= raw_scale
            p4_mass *= raw_scale
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            print(f"Jet[{i}] Raw     Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L1RC
            if level >= CorrectionLevel.L1RC:
                c1 = self.corrL1.evaluate(
                    p4_pt,
                    eta,
                    tree.Jet_area[i],
                    tree.fixedGridRhoFastjetAll
                )
                p4_pt   *= c1
                p4_mass *= c1
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L1Rc    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L2Rel
            if level >= CorrectionLevel.L2REL:
                c2 = self.corrL2.evaluate(
                    p4_pt,
                    eta
                )
                p4_pt   *= c2
                p4_mass *= c2
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L2Rel   Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — L2ResL3Res (data only)
            if self.isData and level >= CorrectionLevel.L2RES_L3RES:
                cr = self.corrL2ResL3.evaluate(
                    p4_pt,
                    eta
                )
                p4_pt   *= cr
                p4_mass *= cr
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] L2ResL3 Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — JER smearing (MC only)
            if self.isMC:
                reso = self.corrJerReso.evaluate(
                    p4_pt,
                    eta,
                    tree.fixedGridRhoFastjetAll
                )
                sf = self.corrJerSf.evaluate(
                    eta,
                    "nom"
                )
                self.randomGen.seed(tree.event + tree.run + tree.luminosityBlock)
                smear = max(
                    0.0,
                    1.0 + self.randomGen.gauss(0.0, reso)
                            * math.sqrt(max(sf*sf - 1.0, 0.0))
                )
                p4_pt   *= smear
                p4_mass *= smear
                p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
                print(f"Jet[{i}] Jer     Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            # — Final Corr
            p4.SetPtEtaPhiM(p4_pt, eta, phi, p4_mass)
            print(f"Jet[{i}] Corr    Pt={p4.Pt():.3f}  M={p4.M():.3f}")

            met -= p4

        print(f"\n[MET Corr]    Pt = {met.Pt():.3f}  Phi = {met.Phi():.3f}\n")
        tree.ChsMET_pt  = met.Pt()
        tree.ChsMET_phi = met.Phi()


def main(input_file: str):
    nano = NanoTree()
    nano.load_tree(input_file)

    scale = ScaleJetMet()
    n_events = nano.get_entries()
    print(f"Events: {n_events}\n")

    for i in range(min(n_events, 10)):
        nano.get_entry(i)
        scale.apply_corrections(nano, CorrectionLevel.L2RES_L3RES)
        print("="*20 + f" End of Event {i} " + "="*20 + "\n")

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "NanoAod.root"
    main(infile)

