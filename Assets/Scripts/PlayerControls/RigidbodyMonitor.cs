using System;
using UnityEngine;

[DisallowMultipleComponent]
public class RigidbodyMonitor : MonoBehaviour
{
    public Rigidbody target;
    public bool enabledLogging = true;
    [Tooltip("Only log when velocity magnitude change exceeds this threshold")]
    public float velocityChangeThreshold = 0.5f; // raised default to avoid noisy per-frame logs

    // Minimum damping change magnitude to log (was ~1e-4 before -> too noisy)
    [Tooltip("Minimum absolute change in linearDamping to trigger a log")]
    public float dampingChangeThreshold = 0.1f;

    // Minimum time between logs for each property to avoid spamming during rapid expected toggles
    [Tooltip("Minimum seconds between successive logs of the same property")]
    public float minLogInterval = 0.2f; // increased default to give aggregation a chance

    Vector3 lastLinearVel;
    float lastLinearDamping;

    // time trackers
    float lastVelLogTime = -10f;
    float lastDampingLogTime = -10f;

    // accumulation for quick successive changes
    int velAccumCount = 0;
    Vector3 velAccumFirst;
    Vector3 velAccumLast;

    int dampingAccumCount = 0;
    float dampingAccumFirst;
    float dampingAccumLast;

    void Reset()
    {
        target = GetComponent<Rigidbody>();
    }

    void Start()
    {
        if (target == null) target = GetComponent<Rigidbody>();
        if (target == null)
        {
            UnityEngine.Debug.LogWarning("[RigidbodyMonitor] No Rigidbody assigned or found.");
            enabledLogging = false;
            return;
        }
        lastLinearVel = target.linearVelocity;
        lastLinearDamping = target.linearDamping;
    }

    void FixedUpdate()
    {
        if (!enabledLogging || target == null) return;

        var v = target.linearVelocity;
        var now = Time.time;

        // Use change in magnitude to reduce noise from small per-component jitter.
        var deltaMag = Math.Abs(v.magnitude - lastLinearVel.magnitude);
        if (deltaMag > velocityChangeThreshold)
        {
            // If enough time passed, emit a log. Otherwise aggregate rapid changes and emit a single summary later.
            if (now - lastVelLogTime >= minLogInterval)
            {
                if (velAccumCount <= 1)
                {
                    UnityEngine.Debug.LogWarning($"[RigidbodyMonitor] linearVelocity changed from {lastLinearVel} -> {v}\nStack:\n{new System.Diagnostics.StackTrace(1, true)}");
                }
                else
                {
                    UnityEngine.Debug.LogWarning($"[RigidbodyMonitor] linearVelocity changed {velAccumCount} times between {velAccumFirst} -> {velAccumLast} (last observed {v})\nStack:\n{new System.Diagnostics.StackTrace(1, true)}");
                }

                velAccumCount = 0;
                lastLinearVel = v;
                lastVelLogTime = now;
            }
            else
            {
                // accumulate quick successive changes and update cached value so baseline stays current
                if (velAccumCount == 0)
                {
                    velAccumFirst = lastLinearVel;
                }
                velAccumLast = v;
                velAccumCount++;
                lastLinearVel = v;
            }
        }
            
        var d = target.linearDamping;
        if (Math.Abs(d - lastLinearDamping) > dampingChangeThreshold)
        {
            if (now - lastDampingLogTime >= minLogInterval)
            {
                if (dampingAccumCount <= 1)
                {
                    UnityEngine.Debug.LogWarning($"[RigidbodyMonitor] linearDamping changed from {lastLinearDamping} -> {d}\nStack:\n{new System.Diagnostics.StackTrace(1, true)}");
                }
                else
                {
                    UnityEngine.Debug.LogWarning($"[RigidbodyMonitor] linearDamping changed {dampingAccumCount} times between {dampingAccumFirst} -> {dampingAccumLast} (last observed {d})\nStack:\n{new System.Diagnostics.StackTrace(1, true)}");
                }

                dampingAccumCount = 0;
                lastLinearDamping = d;
                lastDampingLogTime = now;
            }
            else
            {
                // accumulate
                if (dampingAccumCount == 0) dampingAccumFirst = lastLinearDamping;
                dampingAccumLast = d;
                dampingAccumCount++;
                lastLinearDamping = d;
            }
        }
    }
}