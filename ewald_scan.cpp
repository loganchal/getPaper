#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr int D = 8;
constexpr int NPTS = 6561;
constexpr int WORDS = (NPTS + 63) / 64;
using Point = std::array<int8_t, D>;
using Normal = std::array<int16_t, D>;
using Mask = std::array<uint64_t, WORDS>;
std::array<Point, NPTS> points;

struct NormalHash {
    std::size_t operator()(Normal const& a) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        for (int16_t x : a) {
            uint16_t y = static_cast<uint16_t>(x);
            h ^= static_cast<uint8_t>(y & 255U); h *= 1099511628211ULL;
            h ^= static_cast<uint8_t>(y >> 8); h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};
std::unordered_map<Normal, Mask, NormalHash> mask_cache;

void init_points() {
    for (int idx = 0; idx < NPTS; ++idx) {
        int q = idx;
        for (int j = 0; j < D; ++j) {
            points[idx][j] = static_cast<int8_t>((q % 3) - 1);
            q /= 3;
        }
    }
}

Normal canonical_normal(Normal a) {
    for (int j = 0; j < D; ++j) {
        if (a[j] != 0) {
            if (a[j] < 0) for (auto &x : a) x = static_cast<int16_t>(-x);
            break;
        }
    }
    return a;
}

bool unit_normal(Normal const& a) {
    int nz = 0;
    for (int x : a) {
        if (x != 0) {
            if (std::abs(x) != 1) return false;
            ++nz;
        }
    }
    return nz == 1;
}

Mask make_mask(Normal const& a) {
    Mask out{};
    for (int idx = 0; idx < NPTS; ++idx) {
        int s = 0;
        for (int j = 0; j < D; ++j) s += static_cast<int>(a[j]) * static_cast<int>(points[idx][j]);
        if (std::abs(s) <= 1) out[idx >> 6] |= 1ULL << (idx & 63);
    }
    return out;
}

Mask const& get_mask(Normal a) {
    a = canonical_normal(a);
    auto it = mask_cache.find(a);
    if (it != mask_cache.end()) return it->second;
    auto [pos, inserted] = mask_cache.emplace(a, make_mask(a));
    (void)inserted;
    return pos->second;
}

int popcount_mask(Mask const& m) {
    int n = 0;
    for (uint64_t w : m) n += std::popcount(w);
    return n;
}

std::vector<int> mask_indices(Mask const& m, bool representatives_only) {
    std::vector<int> out;
    out.reserve(popcount_mask(m));
    for (int wi = 0; wi < WORDS; ++wi) {
        uint64_t w = m[wi];
        while (w) {
            int b = std::countr_zero(w);
            int idx = (wi << 6) + b;
            w &= w - 1;
            if (idx >= NPTS) continue;
            if (representatives_only) {
                bool nonzero = false, positive_first = false;
                for (int j = 0; j < D; ++j) {
                    if (points[idx][j] != 0) {
                        nonzero = true;
                        positive_first = points[idx][j] > 0;
                        break;
                    }
                }
                if (!nonzero || !positive_first) continue;
            }
            out.push_back(idx);
        }
    }
    return out;
}

int rank_mod2(Mask const& m) {
    std::array<uint8_t, D> piv{};
    int rank = 0;
    for (int wi = 0; wi < WORDS && rank < D; ++wi) {
        uint64_t w = m[wi];
        while (w && rank < D) {
            int b = std::countr_zero(w);
            int idx = (wi << 6) + b;
            w &= w - 1;
            if (idx >= NPTS) continue;
            uint8_t x = 0;
            for (int j = 0; j < D; ++j) if (points[idx][j] != 0) x |= static_cast<uint8_t>(1U << j);
            for (int col = D - 1; col >= 0 && x; --col) {
                if (!(x & (1U << col))) continue;
                if (piv[col]) x ^= piv[col];
                else { piv[col] = x; ++rank; break; }
            }
        }
    }
    return rank;
}

int rank_modp(Mask const& m, int p) {
    std::array<std::array<int, D>, D> piv{};
    std::array<bool, D> used{};
    int rank = 0;
    auto inv = [p](int a) {
        a %= p; if (a < 0) a += p;
        for (int b = 1; b < p; ++b) if ((a * b) % p == 1) return b;
        return 0;
    };
    for (int wi = 0; wi < WORDS && rank < D; ++wi) {
        uint64_t w = m[wi];
        while (w && rank < D) {
            int b = std::countr_zero(w);
            int idx = (wi << 6) + b;
            w &= w - 1;
            if (idx >= NPTS) continue;
            std::array<int, D> v{};
            for (int j = 0; j < D; ++j) { int z = points[idx][j] % p; if (z < 0) z += p; v[j] = z; }
            for (int col = D - 1; col >= 0; --col) {
                if (v[col] == 0) continue;
                if (used[col]) {
                    int f = v[col];
                    for (int j = 0; j < D; ++j) { v[j] = (v[j] - f * piv[col][j]) % p; if (v[j] < 0) v[j] += p; }
                } else {
                    int f = inv(v[col]);
                    for (int j = 0; j < D; ++j) v[j] = (v[j] * f) % p;
                    piv[col] = v; used[col] = true; ++rank; break;
                }
            }
        }
    }
    return rank;
}

long long det8(std::array<int, D * D> a) {
    long long sign = 1, prev = 1;
    for (int k = 0; k < D - 1; ++k) {
        int pivot = k;
        while (pivot < D && a[pivot * D + k] == 0) ++pivot;
        if (pivot == D) return 0;
        if (pivot != k) {
            for (int j = k; j < D; ++j) std::swap(a[k * D + j], a[pivot * D + j]);
            sign = -sign;
        }
        long long pk = a[k * D + k];
        for (int i = k + 1; i < D; ++i) {
            for (int j = k + 1; j < D; ++j) {
                long long num = static_cast<long long>(a[i * D + j]) * pk - static_cast<long long>(a[i * D + k]) * a[k * D + j];
                if (k > 0) num /= prev;
                a[i * D + j] = static_cast<int>(num);
            }
        }
        prev = pk;
        for (int i = k + 1; i < D; ++i) a[i * D + k] = 0;
    }
    return sign * a[(D - 1) * D + (D - 1)];
}

long long determinant_of_indices(std::array<int, D> const& cols) {
    std::array<int, D * D> a{};
    for (int j = 0; j < D; ++j) for (int i = 0; i < D; ++i) a[i * D + j] = points[cols[j]][i];
    return det8(a);
}

int rank_selected_mod(std::vector<int> const& cols, int candidate = -1) {
    constexpr int p = 1000003;
    std::array<std::array<int, D>, D> rows{};
    int n = static_cast<int>(cols.size()) + (candidate >= 0 ? 1 : 0);
    for (int c = 0; c < n; ++c) {
        int idx = c < static_cast<int>(cols.size()) ? cols[c] : candidate;
        for (int r = 0; r < D; ++r) { int x = points[idx][r]; if (x < 0) x += p; rows[r][c] = x; }
    }
    int rank = 0;
    for (int c = 0; c < n && rank < D; ++c) {
        int pr = rank;
        while (pr < D && rows[pr][c] == 0) ++pr;
        if (pr == D) continue;
        std::swap(rows[rank], rows[pr]);
        long long a = rows[rank][c], inv = 1;
        long long base = a, e = p - 2;
        while (e) { if (e & 1) inv = inv * base % p; base = base * base % p; e >>= 1; }
        for (int j = c; j < n; ++j) rows[rank][j] = static_cast<int>(rows[rank][j] * inv % p);
        for (int r = 0; r < D; ++r) if (r != rank && rows[r][c]) {
            int f = rows[r][c];
            for (int j = c; j < n; ++j) { rows[r][j] = static_cast<int>((rows[r][j] - static_cast<long long>(f) * rows[rank][j]) % p); if (rows[r][j] < 0) rows[r][j] += p; }
        }
        ++rank;
    }
    return rank;
}

struct BasisResult {
    bool found = false;
    long long best_det = std::numeric_limits<long long>::max();
    std::array<int, D> basis{};
};

BasisResult find_basis_heuristic(Mask const& m, uint64_t seed, int attempts = 80) {
    BasisResult res;
    auto reps = mask_indices(m, true);
    std::sort(reps.begin(), reps.end(), [](int a, int b) {
        int sa = 0, sb = 0;
        for (int j = 0; j < D; ++j) { sa += points[a][j] != 0; sb += points[b][j] != 0; }
        if (sa != sb) return sa < sb;
        return a < b;
    });
    std::mt19937_64 rng(seed);
    for (int attempt = 0; attempt < attempts; ++attempt) {
        if (attempt > 0) std::shuffle(reps.begin(), reps.end(), rng);
        std::vector<int> chosen;
        chosen.reserve(D);
        int current_rank = 0;
        for (int idx : reps) {
            int r = rank_selected_mod(chosen, idx);
            if (r > current_rank) {
                chosen.push_back(idx); current_rank = r;
                if (current_rank == D) break;
            }
        }
        if (chosen.size() != D) continue;
        std::array<int, D> cols{};
        std::copy(chosen.begin(), chosen.end(), cols.begin());
        long long det = determinant_of_indices(cols);
        long long ad = std::llabs(det);
        if (ad > 0 && ad < res.best_det) { res.best_det = ad; res.basis = cols; }
        if (ad == 1) { res.found = true; return res; }
        bool improved = true;
        while (improved && ad > 1) {
            improved = false;
            for (int j = 0; j < D && !improved; ++j) {
                int old = cols[j];
                for (int idx : reps) {
                    cols[j] = idx;
                    long long nd = std::llabs(determinant_of_indices(cols));
                    if (nd > 0 && nd < ad) {
                        ad = nd; improved = true;
                        if (ad < res.best_det) { res.best_det = ad; res.basis = cols; }
                        break;
                    }
                }
                if (!improved) cols[j] = old;
            }
        }
        if (ad == 1) { res.found = true; res.basis = cols; return res; }
    }
    return res;
}

std::string point_string(int idx) {
    std::ostringstream os;
    os << '[';
    for (int j = 0; j < D; ++j) { if (j) os << ','; os << static_cast<int>(points[idx][j]); }
    os << ']';
    return os.str();
}

void write_record(std::ostream& os, long long index, std::vector<Normal> const& normals, Mask const& emask,
                  int r2, int r3, int r5, BasisResult const& br, std::string const& kind) {
    os << "BEGIN_CANDIDATE\n";
    os << "kind " << kind << "\nindex " << index << "\ndimension " << D << "\nfacets " << normals.size() << "\n";
    os << "E_count " << popcount_mask(emask) << "\nrank_mod_2 " << r2 << "\nrank_mod_3 " << r3 << "\nrank_mod_5 " << r5 << "\n";
    os << "best_heuristic_det " << (br.best_det == std::numeric_limits<long long>::max() ? -1 : br.best_det) << "\n";
    os << "NORMALS\n";
    for (auto const& u : normals) { os << '1'; for (int x : u) os << ' ' << x; os << '\n'; }
    os << "E_POINTS\n";
    for (int idx : mask_indices(emask, false)) os << point_string(idx) << '\n';
    if (br.found) {
        os << "BASIS\n";
        for (int idx : br.basis) os << point_string(idx) << '\n';
    }
    os << "END_CANDIDATE\n";
}

struct Stats {
    long long total = 0, identity = 0, nonidentity = 0, heuristic = 0, unresolved = 0;
    long long rankdef2 = 0, rankdef3 = 0, rankdef5 = 0;
    int min_E = NPTS + 1;
    long long min_E_index = -1;
};

void process_record(std::vector<std::array<int, D + 1>> const& rows, long long index, Stats& stats,
                    std::ofstream& candidates, int heuristic_attempts) {
    ++stats.total;
    std::vector<Normal> normals;
    normals.reserve(rows.size());
    std::array<bool, D> has_minus_unit{};
    bool identity_ok = true;
    for (auto const& row : rows) {
        if (row[0] != 1) throw std::runtime_error("facet constant is not 1");
        Normal u{};
        for (int j = 0; j < D; ++j) {
            u[j] = static_cast<int16_t>(row[j + 1]);
            if (std::abs(row[j + 1]) > 1) identity_ok = false;
        }
        for (int j = 0; j < D; ++j) {
            bool is = u[j] == -1;
            for (int k = 0; k < D; ++k) if (k != j && u[k] != 0) is = false;
            if (is) has_minus_unit[j] = true;
        }
        normals.push_back(u);
    }
    for (bool x : has_minus_unit) if (!x) throw std::runtime_error("standard -e_i facet normal missing");
    if (identity_ok) { ++stats.identity; return; }
    ++stats.nonidentity;

    Mask emask{};
    emask.fill(~0ULL);
    if (NPTS % 64) emask.back() &= (1ULL << (NPTS % 64)) - 1ULL;
    for (Normal const& u : normals) {
        if (unit_normal(u)) continue;
        Mask const& um = get_mask(u);
        for (int w = 0; w < WORDS; ++w) emask[w] &= um[w];
    }
    int ec = popcount_mask(emask);
    if (ec < stats.min_E) { stats.min_E = ec; stats.min_E_index = index; }
    int r2 = rank_mod2(emask), r3 = rank_modp(emask, 3), r5 = rank_modp(emask, 5);
    bool rankdef = false;
    if (r2 < D) { ++stats.rankdef2; rankdef = true; }
    if (r3 < D) { ++stats.rankdef3; rankdef = true; }
    if (r5 < D) { ++stats.rankdef5; rankdef = true; }
    BasisResult br;
    if (!rankdef) br = find_basis_heuristic(emask, 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(index), heuristic_attempts);
    if (br.found) ++stats.heuristic;
    else {
        ++stats.unresolved;
        write_record(candidates, index, normals, emask, r2, r3, r5, br, rankdef ? "modular_rank_deficiency" : "heuristic_unresolved");
        candidates.flush();
        std::cerr << "UNRESOLVED index=" << index << " facets=" << rows.size() << " E=" << ec
                  << " ranks=" << r2 << ',' << r3 << ',' << r5 << " bestdet="
                  << (br.best_det == std::numeric_limits<long long>::max() ? -1 : br.best_det) << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    long long limit = -1;
    int heuristic_attempts = 80;
    if (argc > 1) limit = std::stoll(argv[1]);
    if (argc > 2) heuristic_attempts = std::stoi(argv[2]);
    init_points();
    mask_cache.reserve(100000);
    std::ofstream candidates("ewald_candidates.txt");
    if (!candidates) { std::cerr << "cannot open candidate output\n"; return 2; }

    Stats stats;
    std::vector<std::array<int, D + 1>> rows;
    bool in_record = false;
    std::string line;
    auto start = std::chrono::steady_clock::now();
    try {
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "FACETS") {
                if (in_record || !rows.empty()) throw std::runtime_error("unexpected FACETS header");
                in_record = true;
            } else if (line.empty()) {
                if (in_record && !rows.empty()) {
                    long long idx = stats.total;
                    process_record(rows, idx, stats, candidates, heuristic_attempts);
                    rows.clear(); in_record = false;
                    if (stats.total % 10000 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double sec = std::chrono::duration<double>(now - start).count();
                        std::cerr << "PROGRESS total=" << stats.total << " identity=" << stats.identity
                                  << " nonidentity=" << stats.nonidentity << " unresolved=" << stats.unresolved
                                  << " masks=" << mask_cache.size() << " minE=" << stats.min_E
                                  << " seconds=" << sec << '\n';
                    }
                    if (limit >= 0 && stats.total >= limit) break;
                }
            } else {
                if (!in_record) throw std::runtime_error("numeric row outside FACETS record: " + line);
                std::istringstream is(line);
                std::array<int, D + 1> row{};
                for (int j = 0; j < D + 1; ++j) if (!(is >> row[j])) throw std::runtime_error("bad facet row: " + line);
                int extra;
                if (is >> extra) throw std::runtime_error("too many entries in facet row: " + line);
                rows.push_back(row);
            }
        }
        if (in_record && !rows.empty() && (limit < 0 || stats.total < limit)) process_record(rows, stats.total, stats, candidates, heuristic_attempts);
    } catch (std::exception const& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 3;
    }
    auto stop = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(stop - start).count();
    std::ofstream summary("ewald_scan_summary.json");
    summary << "{\n"
            << "  \"dimension\": 8,\n"
            << "  \"total\": " << stats.total << ",\n"
            << "  \"identity_basis\": " << stats.identity << ",\n"
            << "  \"nonidentity\": " << stats.nonidentity << ",\n"
            << "  \"heuristic_basis\": " << stats.heuristic << ",\n"
            << "  \"unresolved\": " << stats.unresolved << ",\n"
            << "  \"rank_deficient_mod_2\": " << stats.rankdef2 << ",\n"
            << "  \"rank_deficient_mod_3\": " << stats.rankdef3 << ",\n"
            << "  \"rank_deficient_mod_5\": " << stats.rankdef5 << ",\n"
            << "  \"minimum_E_count\": " << stats.min_E << ",\n"
            << "  \"minimum_E_index\": " << stats.min_E_index << ",\n"
            << "  \"distinct_constraint_masks\": " << mask_cache.size() << ",\n"
            << "  \"seconds\": " << seconds << "\n"
            << "}\n";
    std::cerr << "DONE total=" << stats.total << " identity=" << stats.identity << " nonidentity=" << stats.nonidentity
              << " heuristic=" << stats.heuristic << " unresolved=" << stats.unresolved
              << " rankdef2=" << stats.rankdef2 << " rankdef3=" << stats.rankdef3 << " rankdef5=" << stats.rankdef5
              << " minE=" << stats.min_E << " minEindex=" << stats.min_E_index
              << " masks=" << mask_cache.size() << " seconds=" << seconds << '\n';
    return 0;
}
