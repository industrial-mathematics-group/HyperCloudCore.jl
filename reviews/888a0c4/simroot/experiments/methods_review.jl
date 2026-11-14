# Methods of the 2026-10-26 setup (HyperCloudCore ≥ 0.7.3), after the paper review:
#   - SSP time steppers at every order (RK2/RK3 are SSP; K ≥ 4 uses :RK4SSP, the SSPRK(5,4) of Spiteri–Ruuth),
#   - the AlignedRusanov flux (dissipation oriented by the divergence weights where an order was dropped, so the
#     first-order fallback is monotone for any divergence degree) or the plain Rusanov flux as control,
#   - the MLS (divergence) order frozen at 2 (the review's candidate default), frozen at the scheme order, or
#     coupled to the MOOD order (0),
#   - MOOD U2 with δ = 1e-4 Δx^D (`MOOD_delta_relax`; the strict δ = 0 flags round-off in uniform states).
# Labels: "<stepper>MUSCL<K>(<variant>,<AR|RU>,MLS<m>)", e.g. "RK4SSPMUSCL4(EPD2,AR,MLS2)", variant ∈ none, EPD1,
# EPD2, VK, minmod (1-D only). Included by the experiment files: `add_review_methods!(methods)`.
const REVIEW_DELTA = 1e-4
review_stepper(K) = K <= 2 ? :RK2 : K == 3 ? :RK3 : :RK4SSP
review_label(K, variant, flux, mls) = "$(review_stepper(K))MUSCL$(K)($(variant),$(flux == :AlignedRusanov ? "AR" : "RU"),MLS$(mls))"

function add_review_methods!(methods; orders = 1:5, variants = (:none, :EPD1, :EPD2, :VK, :minmod),
                             fluxes = (:AlignedRusanov, :Rusanov))
    for K in orders, flux in fluxes, variant in variants
        K == 1 && variant != :none && continue
        for mls in unique((0, 2, K))
            K == 1 && mls != 1 && continue
            kw = Dict{Symbol, Any}(:main_flux => flux, :Scheme_MLS_order => mls, :ignore => HyperCloud.ignore_relax,
                                   :custom_label => review_label(K, variant, flux, mls))
            if variant in (:EPD1, :EPD2)
                kw[:mood_criterion] = :U2; kw[:mood_strategy] = variant; kw[:MOOD_delta_relax] = REVIEW_DELTA
            elseif variant == :VK
                kw[:limiter] = :VK; kw[:limiter_mode] = :hard
            elseif variant == :minmod
                kw[:limiter] = :minmod; kw[:limiter_mode] = :hard
            end
            mls == 0 && variant in (:none, :VK, :minmod) && continue   # the coupled order needs MOOD drops
            push!(methods, create_method(review_stepper(K), :MUSCL, K; kw...))
        end
    end
    return methods
end
