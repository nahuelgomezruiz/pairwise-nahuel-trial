#!/bin/bash
# Simple monitoring script to show optimization progress

while true; do
    clear
    echo "==============================================="
    echo "    OPTIMIZATION MONITOR - $(date '+%H:%M:%S')"
    echo "==============================================="
    echo ""
    
    # Check if processes are running
    PROCESS_COUNT=$(ps aux | grep -E "(optimize|pairwise|restart_opt)" | grep -v grep | wc -l | tr -d ' ')
    
    if [ "$PROCESS_COUNT" -gt 0 ]; then
        echo "✅ STATUS: RUNNING ($PROCESS_COUNT active processes)"
        echo ""
        
        # Show what's currently being processed
        echo "📊 CURRENT ACTIVITY:"
        ps aux | grep -E "(optimize_grading)" | grep -v grep | head -1 | sed 's/.*cluster_samples/  Working on: /' | sed 's/.csv.*//' | sed 's/_sample//'
        
        # Show CPU usage
        echo ""
        echo "⚡ CPU USAGE:"
        ps aux | grep -E "(optimize|pairwise)" | grep -v grep | awk '{print "  " $11 ": " $3 "%"}' | sed 's/.*\///' | head -3
        
    else
        echo "⏸️  STATUS: No active processes"
    fi
    
    echo ""
    echo "📁 COMPLETED OPTIMIZATIONS:"
    ls src/data/cluster_samples/*_optimized.csv 2>/dev/null | wc -l | xargs -I {} echo "  {} clusters optimized"
    
    echo ""
    echo "📜 LATEST ACTIVITY:"
    ls -lt *.log 2>/dev/null | head -3 | awk '{print "  " $6 " " $7 " " $8 " - " $9}'
    
    echo ""
    echo "-----------------------------------------------"
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done