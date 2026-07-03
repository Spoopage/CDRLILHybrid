using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

/// <summary>
/// Runtime diagnostic helper: logs Agents, BehaviorParameters, DecisionRequester and DemonstrationRecorder counts.
/// Attach this to a manager GameObject while debugging.
/// </summary>
public class AgentDiagnostics : MonoBehaviour
{
    public KeyCode dumpKey = KeyCode.F12;
    public bool autoDumpOnStart = true;

    void Start()
    {
        if (autoDumpOnStart) DumpAgentInfo();
    }

    void Update()
    {
        if (Input.GetKeyDown(dumpKey))
            DumpAgentInfo();
    }

    public void DumpAgentInfo()
    {
#if UNITY_2023_2_OR_NEWER
        // Use the new faster API and avoid the obsolete warning
        var agents = UnityEngine.Object.FindObjectsByType<Agent>(UnityEngine.FindObjectsSortMode.None);
#else
        // Fallback for older Unity versions
        var agents = FindObjectsOfType<Agent>();
#endif
        Debug.Log($"[AgentDiagnostics] Found {agents.Length} Agent(s) in scene.");

        // Try to resolve DemonstrationRecorder type (may be missing if demonstrations package is not present)
        Type demoType = Type.GetType("Unity.MLAgents.Demonstrations.DemonstrationRecorder, Unity.MLAgents")
                        ?? Type.GetType("Unity.MLAgents.Demonstrations.DemonstrationRecorder");

        for (int i = 0; i < agents.Length; i++)
        {
            var a = agents[i];
            var bp = a.GetComponent<BehaviorParameters>();
            var dr = a.GetComponent<Unity.MLAgents.DecisionRequester>();
            var demo = demoType != null ? a.GetComponent(demoType) : null;
            string bpInfo = bp != null ? $"BehaviorName='{bp.BehaviorName}' Type={bp.BehaviorType} ModelAssigned={(bp.Model != null)}" : "No BehaviorParameters";
            Debug.Log($"[AgentDiagnostics] Agent[{i}] name='{a.gameObject.name}' enabled={a.enabled} activeInHierarchy={a.gameObject.activeInHierarchy} {bpInfo} DecisionRequester={(dr != null)} DemonstrationRecorder={(demo != null)}");
        }

        Debug.Log($"[AgentDiagnostics] Academy.IsCommunicatorOn={Academy.Instance.IsCommunicatorOn}");
    }
}

public class FindAllScripts : MonoBehaviour
{
    void Start()
    {
        MonoBehaviour[] all = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
        foreach (var mb in all)
        {
            if (mb.enabled && mb.gameObject.activeInHierarchy)
                Debug.Log($"Active script: {mb.GetType().Name} on {mb.gameObject.name}");
        }
    }
}