using Unity.MLAgents;
using UnityEngine;
using System.IO;

public class InferenceController : MonoBehaviour
{
    public enum TrackedAgentType { ILAgent, CDRLAgent, HybridAgent }

    [Header("Inference Settings")]
    public int maxEpisodes = 5;
    public float timeScale = 10f;

    [Header("Agent Selection")]
    [Tooltip("Pilih tipe agen yang sedang dijalankan di scene ini.")]
    public TrackedAgentType agentType = TrackedAgentType.ILAgent;

    [Header("Logging")]
    [Tooltip("Nama file CSV log waktu episode. Disimpan di Application.dataPath.")]
    public string logFileName = "InferenceLog_CDRL_MapB.csv";

    private int episodeCount = 0;
    //private bool initialized = false;

    private float sessionStartTime;   // waktu real saat Play ditekan
    private float episodeStartTime;   // waktu real awal episode sekarang

    // Menggunakan base class Agent agar bisa melacak semua tipe agen
    // Tipe konkret dipilih via agentType di Inspector
    //private CDRLAgent trackedAgent;
    //private ILAgent trackedAgent;
    //private HybridAgent trackedAgent;
    private Agent trackedAgent;
    private int lastCompletedEpisodes = 0;

    void Awake()
    {
        Time.timeScale = timeScale;
        Time.fixedDeltaTime = 0.02f /** Time.timeScale*/;
        //Academy.Instance.OnEnvironmentReset += OnEpisodeEnd;

        sessionStartTime = Time.realtimeSinceStartup;
        episodeStartTime = Time.realtimeSinceStartup;

        // Buat file CSV dan tulis header jika belum ada
        string path = GetLogPath();
        if (!File.Exists(path))
        {
            File.WriteAllText(path,
                "Episode,DurasiEpisode_s,WaktuKumulatif_s,Keterangan\n");
        }

        Debug.Log($"[InferenceController] Log akan disimpan ke: {path}");
    }

    void Start()
    {
        // Cari agen di scene berdasarkan tipe yang dipilih di Inspector
        // Dilakukan di Start agar semua Awake sudah selesai
        switch (agentType)
        {
            case TrackedAgentType.CDRLAgent:
                trackedAgent = Object.FindFirstObjectByType<CDRLAgent>();
                break;
            case TrackedAgentType.HybridAgent:
                trackedAgent = Object.FindFirstObjectByType<HybridAgent>();
                break;
            default: // ILAgent
                trackedAgent = Object.FindFirstObjectByType<ILAgent>();
                break;
        }

        if (trackedAgent == null)
            Debug.LogWarning($"[InferenceController] {agentType} tidak ditemukan di scene!");
        else
            Debug.Log($"[InferenceController] {agentType} ditemukan. Mulai memantau episode.");
    }

    void Update()
    {
        if (trackedAgent == null) return;

        // CompletedEpisodes adalah counter resmi ML-Agents, naik 1 setiap EndEpisode()
        int current = trackedAgent.CompletedEpisodes;
        if (current > lastCompletedEpisodes)
        {
            lastCompletedEpisodes = current;
            OnEpisodeEnd();
        }
    }

    private void OnEpisodeEnd()
    {
        // Skip reset pertama saat scene load
        //if (!initialized)
        //{
        //    initialized = true;
        //    episodeStartTime = Time.realtimeSinceStartup;
        //    return;
        //}

        episodeCount++;

        float now = Time.realtimeSinceStartup;
        float episodeDuration = now - episodeStartTime;
        float cumulativeTime = now - sessionStartTime;

        string keterangan = episodeCount >= maxEpisodes ? "SELESAI_SEMUA" : "selesai";

        WriteEpisodeLog(episodeCount, episodeDuration, cumulativeTime, keterangan);

        // Log ke Console
        Debug.Log($"[InferenceController] Episode {episodeCount}/{maxEpisodes} selesai | " +
                  $"Durasi: {episodeDuration:F1}s | " +
                  $"Total: {cumulativeTime:F1}s ({cumulativeTime / 60f:F1} menit)");

        //Debug.Log($"Episode {episodeCount}/{maxEpisodes} selesai. " + $"Real time: {Time.realtimeSinceStartup:F1}s");

        episodeStartTime = now;

        if (episodeCount >= maxEpisodes)
        {
            Debug.Log($"[InferenceController] Semua {maxEpisodes} episode selesai. " +
                      $"Total waktu: {cumulativeTime:F1}s ({cumulativeTime / 60f:F1} menit)");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }

    private string GetLogPath()
    {
        return Path.Combine(Application.dataPath, logFileName);
    }

    private void WriteEpisodeLog(int episode, float duration, float cumulative, string keterangan)
    {
        try
        {
            string path = GetLogPath();
            string row = $"{episode},{duration:F2},{cumulative:F2},{keterangan}\n";
            File.AppendAllText(path, row);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[InferenceController] Gagal menulis log: {e.Message}");
        }
    }

    //void OnDestroy()
    //{
    //    if (Academy.IsInitialized)
    //        Academy.Instance.OnEnvironmentReset -= OnEpisodeEnd;
    //}
}
